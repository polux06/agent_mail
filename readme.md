# Mail Reader Agent

This project contains a Python script that reads emails from an AOL mailbox in read-only mode and summarizes them using OpenAI.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set the following environment variables:

- `EMAIL_ADDRESS` – the AOL email address.
- `EMAIL_PASSWORD` – the application password for the account.
- `OPENAI_API_KEY` – your OpenAI API key.

3. Run the script:

```bash
python mail_reader.py
```

The script keeps track of processed email IDs in `processed_ids.txt` and only processes messages received after 25/06/2025.
