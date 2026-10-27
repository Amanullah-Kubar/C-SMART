from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import os
from dotenv import load_dotenv

from agent.services.memory import get_recent_history, save_message
load_dotenv()
from WhatsApp.messagging import sendMessage


app = FastAPI()

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from agent.config.supabase import GROQ_API_KEY, META_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID
from agent.tools.search_tools import semantic_product_search, get_all_products, check_stock_and_price
from agent.tools.cart_tools import add_to_cart, confirm_order, view_cart
# Register tools
tools = [
    semantic_product_search,
    get_all_products,
    check_stock_and_price,
    add_to_cart,
    view_cart,
    confirm_order,
]
tool_map = {t.name: t for t in tools}

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

system_prompt = SystemMessage(
    content="""# IDENTITY & ROLE
You are "C-SMART", an AI sales assistant for a local retail store in Pakistan. Your goal is to guide shoppers, search inventory accurately, manage their cart, and secure Cash-on-Delivery (COD) checkouts via WhatsApp.

# LANGUAGE & LOCALIZATION GUIDELINES
1. Primary Persona: Conversational, friendly, and respectful Pakistani shop assistant ("Ji bilkul", "Zaroor", "Aap").
2. Language Matching:
   - If the user talks in Roman Urdu or Urdu ("koi keyboard hai?", "kya price hai?"): Respond strictly in natural Roman Urdu.
   - If the user talks in English: Respond in fluent, clear English.
   - If the user uses Hindi vocabulary (e.g., "shukriya/dhanyawad", "mulya", "namaste"): Convert tone to Pakistani Roman Urdu ("Shukriya", "Price/Qeemat").
3. Numbers & Currency: NEVER spell out numbers as words or Roman numerals. Always format currency as standard numerals with PKR (e.g., "PKR 4,500" or "Rs. 4,500").
    - currency symbols: Use "PKR" or "Rs." for Pakistani Rupees, not "$" or "USD".

# TOOL USAGE INSTRUCTIONS
use my supabase strictly for all product lookups, stock checks, and cart management. Do not make up prices or stock quantities. Always quote the exact values returned by the tools.
1. Use `semantic_product_search` for natural language inquiries and feature requests.
2. Use `get_all_products` when the user asks to see the catalog or filter by price/availability.
3. Use `check_stock_and_price` to verify real-time price and stock of a specific item.
4. Use `add_to_cart` when the user wants to add an item to their shopping cart.
5. Use `view_cart` when the user wants to see what is in their cart or check their subtotal.
6. Use `confirm_order` to finalize the order and generate a payment link.

# WHATSAPP FORMATTING CONSTRAINTS
- Messages must be compact and scannable on mobile screens. Keep replies under 3-4 bullet points.
- Use native WhatsApp Markdown:
  * *Bold* for product titles, prices, and status.
  * Bullet points for listing options.
- Avoid large blocks of descriptive text. State the item, price, stock status, and ask a single clear question to advance the sale (e.g., "Kya aap isay cart mein add karna chahein gey?").

# GUARDRAILS
- Out-of-Stock Items: If `stock_quantity` is 0, explicitly warn the customer that it is currently unavailable and suggest an alternative using semantic search.
- Price Integrity: Only quote prices returned directly from tool outputs.
"""
)

def ask_agent(user_query: str, user_id: str = "guest_user") -> str:
    print(f"\n{'='*50}\n👤 User ({user_id}): {user_query}\n{'='*50}")

    # 1. Pehle user ka message DB mein save karein
    save_message(user_id=user_id, role="user", content=user_query)

    # 2. Database se aakhri 4 ya 6 messages uthayein
    past_history = get_recent_history(user_id=user_id, limit=6)

    # 3. Prompt context assemble karein: System Prompt + Past Turns + Current Turn
    # Note: user_query past_history mein already last message ban chuka hoga agar get_recent_history insert ke baad chala
    # Behtar pattern yeh hai:
    messages = [system_prompt] + past_history

    max_steps = 3
    final_reply = "Koi masla agaya hai, please dobara try karein."

    for _ in range(max_steps):
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        # Agar tool nahi chalana, to final text response mil gaya
        if not ai_response.tool_calls:
            final_reply = ai_response.content
            break

        # Tool execution loop
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in ["add_to_cart", "view_cart", "confirm_order"]:
                tool_args["user_id"] = user_id

            if tool_name in tool_map:
                tool_func = tool_map[tool_name]
                tool_output = tool_func.invoke(tool_args)
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."

            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            )
    else:
        # Loop exhaust ho gaya
        final_reply = messages[-1].content if messages[-1].content else "Request process ho gayi hai."

    # 4. Agent ka final reply DB mein save karein taake agle turn pe yaad rahe
    save_message(user_id=user_id, role="assistant", content=final_reply)

    return final_reply

# A secret verification string you define (must match Meta dashboard)
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "my_custom_secret_token_123")

def process_whatsapp_message(payload: dict):
    """Background task to handle LLM routing, db updates, and replies."""
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # Check if incoming event is a message
        if "messages" in value:
            message = value["messages"][0]
            sender_id = message["from"]  # User's phone number
            message_type = message["type"]
            
            if message_type == "text":
                user_text = message["text"]["body"]
                print(f"Received from {sender_id}: {user_text}")
                message = ask_agent(user_query=user_text, user_id=sender_id)
                print(message)
                response = sendMessage(sender_id, message)
                print(f"Sent to {sender_id}: {response}")
                # Pass data to LangChain agent / database lookup
    except Exception as e:
        print(f"Error parsing message payload: {e}")

# 1. Verification Handshake (Meta calls this once during setup)
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        # Meta strictly requires returning the hub.challenge as plain text
        return PlainTextResponse(content=challenge, status_code=200)
    
    raise HTTPException(status_code=403, detail="Verification token mismatch")

# 2. Ingestion Route (Meta sends every message/event here)
@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    # Offload execution to avoid Meta's 3-second timeout window
    background_tasks.add_task(process_whatsapp_message, payload)

    # Immediately respond with HTTP 200 OK
    return {"status": "success"}

from fastapi import Header

# Add this route below your WhatsApp webhooks
@app.post("/webhook/safepay")
async def safepay_webhook(
    request: Request,
    x_sfpy_signature: str = Header(None),
    x_sfpy_timestamp: str = Header(None)
):
    # Insert the Safepay signature verification here
    # Update Supabase order status to "confirmed" based on the metadata order_id
    
    return {"status": "success"}

