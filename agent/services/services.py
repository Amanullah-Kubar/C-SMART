# import numpy as np
# from sentence_transformers import SentenceTransformer
# from types import List

# import httpx
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# from agent.config.supabase import GROQ_API_KEY, META_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID
# from agent.tools.search_tools import semantic_product_search, get_all_products, check_stock_and_price
# from agent.tools.cart_tools import add_to_cart, view_cart, checkout_cart

# # Register tools
# tools = [
#     semantic_product_search,
#     get_all_products,
#     check_stock_and_price,
#     add_to_cart,
#     view_cart,
#     checkout_cart,
# ]
# tool_map = {t.name: t for t in tools}

# # Initialize LLM
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     groq_api_key=GROQ_API_KEY,
#     temperature=0.1
# )
# llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# system_prompt = SystemMessage(
#     content="""You are a helpful e-commerce store assistant.
# You have access to tools that query the store database:
# 1. Use `semantic_product_search` for natural language inquiries and feature requests.
# 2. Use `get_all_products` when the user asks to see the catalog or filter by price/availability.
# 3. Use `check_stock_and_price` to verify real-time price and stock of a specific item.
# 4. Use `add_to_cart` when the user wants to add an item to their shopping cart.
# 5. Use `view_cart` when the user wants to see what is in their cart or check their subtotal.
# 6. Use `checkout_cart` when the user is ready to buy and has provided a delivery address.

# Always answer concisely and accurately based on the tool results.
# always translate the answer in the language of the user query.
# parse the answer into roman langage dont change the numbers into roman numbers and the currency into roman currency.
# always convert hindi to urdu roman language."""
# )

# def ask_agent(user_query: str, user_id: str = "guest_user") -> str:
#     print(f"\n{'='*50}\n👤 User ({user_id}): {user_query}\n{'='*50}")
#     messages = [system_prompt, HumanMessage(content=user_query)]

#     ai_response = llm_with_tools.invoke(messages)
#     messages.append(ai_response)

#     if ai_response.tool_calls:
#         for tool_call in ai_response.tool_calls:
#             tool_name = tool_call["name"]
#             tool_args = tool_call["args"]

#             if tool_name in ["add_to_cart", "view_cart", "checkout_cart"]:
#                 if "user_id" not in tool_args:
#                     tool_args["user_id"] = user_id

#             tool_function = tool_map[tool_name]
#             tool_output = tool_function.invoke(tool_args)
#             messages.append(
#                 ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
#             )

#         final_response = llm.invoke(messages)
#         return final_response.content

#     return ai_response.content

# # Outbound WhatsApp message dispatcher
# async def send_whatsapp_message(to: str, text: str):
#     print(f"Sending WhatsApp message to {to}: {text}")
#     url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
#     headers = {
#         "Authorization": f"Bearer {META_ACCESS_TOKEN}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "messaging_product": "whatsapp",
#         "to": to,
#         "type": "text",
#         "text": {"body": text}
#     }
#     async with httpx.AsyncClient() as client:
#         await client.post(url, headers=headers, json=payload)

# # Background pipeline bridging inbound chat with the agent
# async def handle_whatsapp_message_pipeline(sender_id: str, message_text: str):
#     agent_reply = ask_agent(user_query=message_text, user_id=sender_id)
#     await send_whatsapp_message(to=sender_id, text=agent_reply)


from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self.load_model()

    def load_model(self):
        """Loads sentence transformer model."""
        try:
            # Fix: Assign to self.model
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(
                f"Model '{self.model_name}' loaded successfully. Dimension:"
                f" {self.dimension}"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Converts a list of text strings into a 2D numpy array of embeddings."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        return self.model.encode(texts, show_progress_bar=True)