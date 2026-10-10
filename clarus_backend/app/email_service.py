"""Email delivery for Monitor-category follow-up reminders via Gmail API."""

import base64
from email.message import EmailMessage
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"


def _get_gmail_service():
    credentials = None

    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES,
        )

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES,
        )
        credentials = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def send_followup_reminder(to_email: str, return_date: str) -> bool:
    """
    Send one follow-up reminder through Gmail. Returns True on success;
    callers should not crash the reminder job over one failed address.
    """
    msg = EmailMessage()
    msg["Subject"] = "Clarus Screening - Follow-up Visit Recommended"
    msg["To"] = to_email
    msg.set_content(
        "Your recent diabetic retinopathy screening suggested a follow-up "
        f"visit around {return_date}.\n\n"
        "Please visit your barangay health center at your convenience to "
        "have this checked again.\n\n"
        "This is an automated message from the Clarus screening system. "
        "It is not a diagnosis — please consult your healthcare worker "
        "with any questions."
    )

    try:
        encoded_message = base64.urlsafe_b64encode(
            msg.as_bytes()
        ).decode("utf-8")
        service = _get_gmail_service()
        service.users().messages().send(
            userId="me",
            body={"raw": encoded_message},
        ).execute()
        return True
    except Exception as exc:
        print(f"[email_service] Failed to send to {to_email}: {exc}")
        return False