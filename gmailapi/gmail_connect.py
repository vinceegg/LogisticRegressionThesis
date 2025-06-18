import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import html
import datetime

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate and create the Gmail API service."""
    creds = None
    credentials_path = os.path.join('config', 'credentials1.json')
    token_path = 'token.pickle'

    # Check if token already exists
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If credentials don't exist or are invalid, refresh them or get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=54920)  # Set a fixed port
            except Exception as e:
                print(f"Authentication error: {e}")
                print("Please make sure you've added your email as a test user in the Google Cloud Console")
                return None
        
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build('gmail', 'v1', credentials=creds)
        print("Successfully authenticated with Gmail API")
        return service
    except HttpError as error:
        print(f'Error building service: {error}')
        return None

def fetch_emails(service, user_id='me', max_results=100):
    """Fetch recent emails with more complete data for display in the UI."""
    if not service:
        return []
        
    try:
        response = service.users().messages().list(userId=user_id, maxResults=max_results).execute()
        messages = response.get('messages', [])

        if not messages:
            print("No messages found.")
            return []

        print(f"Fetching {len(messages)} emails...")
        email_list = []
        for msg in messages:
            # Get full message details
            msg_data = service.users().messages().get(userId=user_id, id=msg['id'], format='full').execute()
            
            # Process headers
            headers = msg_data.get('payload', {}).get('headers', [])
            email_info = {
                "id": msg['id'],
                "subject": "No Subject",
                "sender": "Unknown Sender",
                "date": "Unknown Date",
                "snippet": msg_data.get("snippet", "No preview available"),
                "labels": msg_data.get("labelIds", [])
            }
            
            # Extract header information
            for header in headers:
                name = header["name"].lower()
                if name == "subject":
                    email_info["subject"] = header["value"] or "No Subject"
                elif name == "from":
                    email_info["sender"] = header["value"]
                elif name == "date":
                    try:
                        # Parse and format date
                        date_str = header["value"]
                        email_info["date"] = date_str
                        
                        # Could add more sophisticated date parsing here if needed
                    except:
                        email_info["date"] = header["value"]
            
            # Extract message body
            email_info["body"] = extract_email_body(msg_data)
            
            # Check if email is read or unread
            email_info["unread"] = "UNREAD" in email_info["labels"]
            
            # Add to email list
            email_list.append(email_info)

        return email_list

    except HttpError as error:
        print(f'Error fetching emails: {error}')
        return []

def extract_email_body(message):
    """Extract and decode the email body from the message."""
    body = ""
    
    if 'payload' not in message:
        return "No message content"
    
    # Check for plain/text or HTML parts
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                break
            elif part['mimeType'] == 'text/html' and 'data' in part['body']:
                html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                body = html_to_plain_text(html_body)
                break
    
    # If no parts found, check for body directly in the payload
    elif 'body' in message['payload'] and 'data' in message['payload']['body']:
        data = message['payload']['body']['data']
        body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        # If it's HTML, convert to plain text
        if message['payload'].get('mimeType') == 'text/html':
            body = html_to_plain_text(body)
    
    return body if body else "No message content available"

def html_to_plain_text(html_content):
    """Simple conversion of HTML to plain text."""
    # Remove HTML tags - this is a simple approach
    # For more complex HTML parsing, consider using a library like BeautifulSoup
    text = html.unescape(html_content)
    
    # Replace common HTML elements with text equivalents
    replacements = [
        ('<br>', '\n'), ('<br/>', '\n'), ('<br />', '\n'),
        ('<p>', '\n'), ('</p>', '\n'),
        ('<div>', '\n'), ('</div>', '\n'),
        ('<tr>', '\n'), ('</tr>', '\n'),
        ('<td>', ' '), ('</td>', ' '),
        ('<th>', ' '), ('</th>', ' ')
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
        text = text.replace(old.upper(), new)  # Handle uppercase tags too
    
    # Remove remaining HTML tags
    in_tag = False
    plain_text = ""
    for char in text:
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
        elif not in_tag:
            plain_text += char
    
    return plain_text

def get_email_by_id(service, email_id, user_id='me'):
    """Fetch a specific email by ID with full content."""
    if not service:
        return None
        
    try:
        msg_data = service.users().messages().get(userId=user_id, id=email_id, format='full').execute()
        
        # Process headers
        headers = msg_data.get('payload', {}).get('headers', [])
        email_info = {
            "id": email_id,
            "subject": "No Subject",
            "sender": "Unknown Sender",
            "date": "Unknown Date",
            "to": "",
            "cc": "",
            "snippet": msg_data.get("snippet", "No preview available"),
            "labels": msg_data.get("labelIds", [])
        }
        
        # Extract header information
        for header in headers:
            name = header["name"].lower()
            if name == "subject":
                email_info["subject"] = header["value"] or "No Subject"
            elif name == "from":
                email_info["sender"] = header["value"]
            elif name == "date":
                email_info["date"] = header["value"]
            elif name == "to":
                email_info["to"] = header["value"]
            elif name == "cc":
                email_info["cc"] = header["value"]
        
        # Extract message body
        email_info["body"] = extract_email_body(msg_data)
        
        return email_info
        
    except HttpError as error:
        print(f'Error fetching email: {error}')
        return None