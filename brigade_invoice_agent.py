"""
Brigade Invoice Agent
Finds all invoices and documents from Brigade Group in Gmail and Google Drive,
then stores them in the Brigade Calista folder on Google Drive.

Brigade Calista folder ID: 1mdkuhuT4BJ7GutcDEvU0LLz7OwtegVVQ
"""

import os
import io
import base64
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive",
]

BRIGADE_CALISTA_FOLDER_ID = "1mdkuhuT4BJ7GutcDEvU0LLz7OwtegVVQ"

GMAIL_QUERY = (
    "from:brigadegroup.com OR from:brigadecalista.com "
    "has:attachment"
)

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def get_credentials(token_file="token.json", credentials_file="credentials.json"):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as f:
            f.write(creds.to_json())
    return creds


def get_existing_filenames(drive_service, folder_id):
    """Return a set of filenames already in the target Drive folder."""
    existing = set()
    page_token = None
    while True:
        resp = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(name)",
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            existing.add(f["name"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return existing


def list_gmail_threads(gmail_service, query, max_results=200):
    threads = []
    page_token = None
    while len(threads) < max_results:
        resp = gmail_service.users().threads().list(
            userId="me",
            q=query,
            maxResults=min(50, max_results - len(threads)),
            pageToken=page_token,
        ).execute()
        threads.extend(resp.get("threads", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return threads


def get_attachments_from_message(msg):
    """Recursively extract attachments from a message payload."""
    attachments = []
    payload = msg.get("payload", {})

    def walk_parts(parts):
        for part in parts:
            if part.get("parts"):
                walk_parts(part["parts"])
            filename = part.get("filename", "")
            mime_type = part.get("mimeType", "")
            body = part.get("body", {})
            attachment_id = body.get("attachmentId")
            if filename and attachment_id and mime_type in ALLOWED_MIME_TYPES:
                attachments.append({
                    "filename": filename,
                    "mime_type": mime_type,
                    "attachment_id": attachment_id,
                })

    if payload.get("parts"):
        walk_parts(payload["parts"])
    return attachments


def upload_to_drive(drive_service, filename, mime_type, data, folder_id):
    file_metadata = {
        "name": filename,
        "parents": [folder_id],
    }
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name",
    ).execute()
    return uploaded


def run_agent():
    print("Brigade Invoice Agent starting...")
    creds = get_credentials()
    gmail_service = build("gmail", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    print(f"Checking existing files in Brigade Calista folder...")
    existing = get_existing_filenames(drive_service, BRIGADE_CALISTA_FOLDER_ID)
    print(f"  {len(existing)} file(s) already present.")

    print(f"Searching Gmail for Brigade Group emails with attachments...")
    threads = list_gmail_threads(gmail_service, GMAIL_QUERY)
    print(f"  Found {len(threads)} thread(s).")

    uploaded_count = 0
    skipped_count = 0

    for thread in threads:
        thread_data = gmail_service.users().threads().get(
            userId="me",
            id=thread["id"],
            format="full",
        ).execute()

        for msg in thread_data.get("messages", []):
            attachments = get_attachments_from_message(msg)
            for att in attachments:
                filename = att["filename"]
                if filename in existing:
                    print(f"  [SKIP] {filename} (already in Drive)")
                    skipped_count += 1
                    continue

                att_data = gmail_service.users().messages().attachments().get(
                    userId="me",
                    messageId=msg["id"],
                    id=att["attachment_id"],
                ).execute()

                raw_data = base64.urlsafe_b64decode(att_data["data"])
                result = upload_to_drive(
                    drive_service,
                    filename,
                    att["mime_type"],
                    raw_data,
                    BRIGADE_CALISTA_FOLDER_ID,
                )
                print(f"  [UPLOAD] {filename} -> Drive ID: {result['id']}")
                existing.add(filename)
                uploaded_count += 1

    print(f"\nDone. Uploaded: {uploaded_count}, Skipped (duplicates): {skipped_count}")


if __name__ == "__main__":
    run_agent()
