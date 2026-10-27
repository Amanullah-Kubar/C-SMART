import os
from dotenv import load_dotenv
import requests
load_dotenv()
whatsapp_access_token = os.getenv("WHATSAPP_TOKEN")
whatsapp_phone_number_id = os.getenv("PHONE_NUMBER_ID")


def sendMessage(phone_number, message):
    print(whatsapp_access_token)
    print(whatsapp_phone_number_id)

    url = f"https://graph.facebook.com/v18.0/{whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {whatsapp_access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "body": message
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    print("WhatsApp API status:", response.status_code)
    print("WhatsApp API response:", response.json())

    return response.json()