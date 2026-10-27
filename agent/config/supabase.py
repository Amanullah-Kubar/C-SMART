import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

# backend/.env
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

print("SUPABASE_URL:", supabase_url)
print("SUPABASE_KEY loaded:", bool(supabase_key))

supabase: Client = create_client(
    supabase_url,
    supabase_key
)