"""Email delivery for Monitor-category follow-up reminders via Gmail API.

Auth model
----------
Sending uses OAuth 2.0 (scope: gmail.send). Authorization is a ONE-TIME,
interactive step done on a developer machine:

    python -m app.email_service

That opens a browser, you sign in as the sender account, and the resulting
token is saved to token.json. From then on the API only *refreshes* the
token silently. It never opens a browser, so it cannot hang a request or
the scheduler. If the token is missing, expired, or revoked, sending raises
GmailAuthError with instructions for re-authorizing.

Deployment (no browser, possibly no persistent disk): put the full contents
of token.json in the GMAIL_TOKEN_JSON environment variable. The token file
already contains the client id/secret and refresh token needed to refresh.
"""

import base64
import json
import os
from email.message import EmailMessage
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"
TOKEN_ENV_VAR = "GMAIL_TOKEN_JSON"

REAUTH_HINT = (
    "Gmail authorization is missing, expired, or revoked. Re-authorize by "
    "running `python -m app.email_service` on a machine with a browser, then "
    f"deploy the new token.json (or update {TOKEN_ENV_VAR})."
)


class GmailAuthError(RuntimeError):
    """Gmail credentials are missing, expired, or revoked."""


def _load_credentials() -> Credentials | None:
    token_json = os.getenv(TOKEN_ENV_VAR)
    if token_json:
        return Credentials.from_authorized_user_info(
            json.loads(token_json), SCOPES
        )
    if TOKEN_FILE.exists():
        return Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    return None


def _get_gmail_service(interactive: bool = False):
    """
    Build a Gmail API client.

    interactive=False (default, used by the API and scheduler): never opens
    a browser; raises GmailAuthError if credentials can't be used.
    interactive=True (only the one-time `python -m app.email_service`
    command): runs the browser sign-in flow and saves token.json.
    """
    credentials = _load_credentials()

    if credentials and not credentials.valid:
        if credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError as exc:
                if not interactive:
                    raise GmailAuthError(REAUTH_HINT) from exc
                credentials = None
            else:
                # Keep the on-disk token current when file-based.
                if not os.getenv(TOKEN_ENV_VAR):
                    TOKEN_FILE.write_text(
                        credentials.to_json(), encoding="utf-8"
                    )
        else:
            credentials = None

    if not credentials or not credentials.valid:
        if not interactive:
            raise GmailAuthError(REAUTH_HINT)
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES,
        )
        credentials = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def send_followup_reminder(to_email: str, return_date: str) -> bool:
    """
    Send one follow-up reminder through Gmail.

    Returns True on success and False if this one message failed (e.g. a
    bad address), so one failure doesn't stop the batch. Raises
    GmailAuthError when Gmail itself can't be used at all, since every
    remaining message would fail for the same reason.
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

    service = _get_gmail_service()  # may raise GmailAuthError (propagates)

    try:
        encoded_message = base64.urlsafe_b64encode(
            msg.as_bytes()
        ).decode("utf-8")
        service.users().messages().send(
            userId="me",
            body={"raw": encoded_message},
        ).execute()
        return True
    except Exception as exc:
        print(f"[email_service] Failed to send to {to_email}: {exc}")
        return False


if __name__ == "__main__":
    _get_gmail_service(interactive=True)
    print(f"Gmail authorized. Token saved to {TOKEN_FILE}")
