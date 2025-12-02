#!/usr/bin/env python3
"""Send generated email via SMTP."""

from __future__ import annotations

import argparse
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send weekly review email.")
    parser.add_argument(
        "--email-html",
        default="layer-5-email-generation/output/latest/email.html",
        help="Path to HTML email file.",
    )
    parser.add_argument(
        "--to",
        required=True,
        help="Recipient email address.",
    )
    parser.add_argument(
        "--subject",
        default="Weekly Review Pulse - INDmoney",
        help="Email subject line.",
    )
    parser.add_argument(
        "--from-email",
        default=None,
        help="Sender email address (defaults to SMTP_USER env var).",
    )
    parser.add_argument(
        "--smtp-user",
        default=None,
        help="SMTP username (defaults to SMTP_USER env var).",
    )
    parser.add_argument(
        "--smtp-password",
        default=None,
        help="SMTP password (defaults to SMTP_PASSWORD env var).",
    )
    parser.add_argument(
        "--smtp-host",
        default="smtp.gmail.com",
        help="SMTP server hostname.",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=587,
        help="SMTP server port.",
    )
    return parser.parse_args()


def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    from_email: str,
    smtp_user: str,
    smtp_password: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> bool:
    """Send email using smtplib."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)

        print(f"Connecting to {smtp_host}:{smtp_port}...")
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            print("Logging in...")
            server.login(smtp_user, smtp_password)
            print("Sending email...")
            server.send_message(msg)
        
        print(f"✓ Email sent successfully to {to_email}")
        return True
    except Exception as exc:
        print(f"✗ Failed to send email: {exc}")
        return False


def main() -> None:
    args = parse_args()
    
    email_path = Path(args.email_html)
    if not email_path.exists():
        print(f"✗ Email file not found: {email_path}")
        print("\nPlease generate the email first:")
        print("  python layer-5-email-generation/generate_email.py")
        return
    
    html_content = email_path.read_text(encoding="utf-8")
    
    # Get SMTP credentials from args or environment
    smtp_user = args.smtp_user or os.environ.get("SMTP_USER")
    smtp_password = args.smtp_password or os.environ.get("SMTP_PASSWORD")
    from_email = args.from_email or smtp_user
    
    if not smtp_user or not smtp_password:
        print("✗ SMTP credentials required.")
        print("\nOptions:")
        print("1. Set environment variables:")
        print("   export SMTP_USER='your-email@gmail.com'")
        print("   export SMTP_PASSWORD='your-app-password'")
        print("\n2. Or pass as arguments:")
        print("   --smtp-user 'your-email@gmail.com' --smtp-password 'your-app-password'")
        print("\nFor Gmail:")
        print("- Enable 2-factor authentication")
        print("- Generate App Password: https://myaccount.google.com/apppasswords")
        print("- Use the 16-character app password (not your regular password)")
        return
    
    success = send_email(
        to_email=args.to,
        subject=args.subject,
        html_content=html_content,
        from_email=from_email,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
