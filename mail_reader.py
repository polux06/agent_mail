import os
import imaplib
import email
from email.header import decode_header
from datetime import datetime
from pathlib import Path
import openai

START_DATE = datetime(2025, 6, 25)
PROCESSED_FILE = Path('processed_ids.txt')

EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not all([EMAIL_ADDRESS, EMAIL_PASSWORD, OPENAI_API_KEY]):
    raise EnvironmentError('EMAIL_ADDRESS, EMAIL_PASSWORD, and OPENAI_API_KEY must be set as environment variables.')

openai.api_key = OPENAI_API_KEY


def load_processed_ids():
    if PROCESSED_FILE.exists():
        return {line.strip() for line in PROCESSED_FILE.read_text().splitlines()}
    return set()


def save_processed_ids(ids):
    PROCESSED_FILE.write_text('\n'.join(ids))


def summarize_email(body):
    prompt = (
        "You are an assistant summarizing an email. Extract the sender, title, urgency, summary, "
        "and any meetings or TODO items. Answer in JSON with keys sender, title, urgency, summary, "
        "meetings, todos."
    )
    completion = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": body}]
    )
    return completion.choices[0].message.content


def main():
    processed = load_processed_ids()
    mail = imaplib.IMAP4_SSL('imap.aol.com')
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select('INBOX', readonly=True)

    start_str = START_DATE.strftime('%d-%b-%Y')
    status, data = mail.search(None, f'(SINCE {start_str})')
    if status != 'OK':
        print('No messages found!')
        return
    ids = data[0].split()
    for msg_id in ids:
        if msg_id.decode() in processed:
            continue
        status, msg_data = mail.fetch(msg_id, '(RFC822)')
        if status != 'OK':
            continue
        raw_email = msg_data[0][1]
        message = email.message_from_bytes(raw_email)
        body = ""
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' and part.get('Content-Disposition') is None:
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            body = message.get_payload(decode=True).decode(errors='ignore')
        summary = summarize_email(body)
        print(f"Summary for email {msg_id.decode()}:")
        print(summary)
        processed.add(msg_id.decode())
    save_processed_ids(processed)
    mail.logout()


if __name__ == '__main__':
    main()
