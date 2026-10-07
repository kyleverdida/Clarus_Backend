"""
Email delivery for Monitor-category follow-up reminders.

Sends via Gmail SMTP using an App Password — never the real account
password, and never committed to the repo (see .env / environment config).
"""

import os
import smtplib
from email.message import EmailMessage

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("CLARUS_EMAIL_USER")
SMTP_PASSWORD = os.getenv("CLARUS_EMAIL_APP_PASSWORD")


def send_followup_reminder(to_email: str, return_date: str) -> bool:
    """
    Sends a single follow-up reminder email. Returns True on success,
    False on failure — callers should not crash the reminder job over
    one bad email address.
    """
    msg = EmailMessage()
    msg["Subject"] = "Clarus Screening — Follow-up Visit Recommended"
    msg["From"] = SMTP_USER
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
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f"[email_service] Failed to send to {to_email}: {exc}")
        return False