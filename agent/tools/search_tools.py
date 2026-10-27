# 1. Initialize Clients
from urllib import response

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from agent.tools.cart_tools import add_to_cart, view_cart, confirm_order
from agent.config.supabase import supabase
import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")


model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# TOOL 1: Semantic Vector Search (For descriptive queries)
# ============================================================
@tool
def semantic_product_search(query: str, top_k: int = 3) -> str:
    """Use this tool to search for specific products by natural language descriptions,

    features, use-cases, synonyms, or vague requests (e.g., 'wireless headphones
    with bass',
    'gadget for typing', 'device to connect HDMI').
    """
    print(f"\n[🔧 Tool Triggered: semantic_product_search(query='{query}')]")

    query_vec = model.encode([query])[0].tolist()

    response = supabase.rpc(
        "match_products",
        {
            "query_embedding": query_vec,
            "match_threshold": -1.0,  # return nearest matches
          "match_count": top_k,
      },
    ).execute()

    products = response.data or []
    if not products:
        return "No matching products found."

    formatted = [
        f"- {p['title']} (${p['price']}): {p['description']} | Stock:"
        f" {p['stock_quantity']}"
        for p in products
    ]
    return "\n".join(formatted)


# ============================================================
# TOOL 2: Structured SQL Catalog Tool (For list all, filters, aggregations)
# ============================================================
@tool
def get_all_products(
    max_price: float = None, in_stock_only: bool = False
) -> str:
    """Use this tool when the user asks to see ALL products, lists the entire catalog,

    or asks for products filtered strictly by maximum price or availability
    without a specific semantic search term.
    """
    print(
        f"\n[🔧 Tool Triggered: get_all_products(max_price={max_price},"
        f" in_stock_only={in_stock_only})]"
    )

    query_builder = supabase.table("products").select(
        "title, price, description, stock_quantity"
    )

    if in_stock_only:
        query_builder = query_builder.gt("stock_quantity", 0)

    if max_price is not None:
        query_builder = query_builder.lte("price", max_price)

    response = query_builder.limit(20).execute()
    products = response.data or []

    if not products:
        return "No products found matching those criteria."

    formatted = [
        f"- {p['title']} (${p['price']}) | Stock: {p['stock_quantity']} units |"
        f" {p['description']}"
        for p in products
    ]
    return "\n".join(formatted)


# ============================================================
# TOOL 3: Real-Time Stock & Price Checker (Exact lookup)
# ============================================================
@tool
def check_stock_and_price(product_title: str) -> str:
    """Use this tool when the user asks specifically about the exact price or available

    stock quantity of a known product name.
    """
    print(
      f"\n[🔧 Tool Triggered: check_stock_and_price(product_title='{product_title}')]"
    )

    response = (
        supabase.table("products")
        .select("title, price, stock_quantity")
        .ilike("title", f"%{product_title}%")
        .execute()
    )

    products = response.data or []
    if not products:
        return (
            f"Product '{product_title}' was not found in the inventory database."
        )

    return "\n".join([
      f"- {p['title']}: Price = ${p['price']}, Current Stock ="
      f" {p['stock_quantity']} units"
      for p in products
    ])

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# 1. Register tools and bind to Groq
tools = [
    semantic_product_search,
    get_all_products,
    check_stock_and_price,
    add_to_cart,
    view_cart,
    confirm_order,
]
tool_map = {t.name: t for t in tools}

llm = ChatGroq(
    model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY, temperature=0.1
)

llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# 2. Agent Execution Loop
system_prompt = SystemMessage(
    content="""You are a helpful e-commerce store assistant.
You have access to tools that query the store database:
1. Use `semantic_product_search` for natural language inquiries and feature requests.
2. Use `get_all_products` when the user asks to see the catalog or filter by price/availability.
3. Use `check_stock_and_price` to verify real-time price and stock of a specific item.
4. Use `add_to_cart` when the user wants to add an item to their shopping cart.
5. Use `view_cart` when the user wants to see what is in their cart or check their subtotal.
6. Use `checkout_cart` when the user is ready to buy and has provided a delivery address.

Always answer concisely and accurately based on the tool results.
always translate the answer in the language of the user query.
parse the answer into roman langage dont change the numbers into roman numbers and the currency into roman currency.
always convert hindi to urdu roman language."""
)


def ask_agent(user_query: str, user_id: str = "guest_user"):
    print(f"\n{'='*50}\n👤 User ({user_id}): {user_query}\n{'='*50}")

    messages = [system_prompt, HumanMessage(content=user_query)]

    # First pass: model decides which tool(s) to call
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if ai_response.tool_calls:
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

        # Automatically inject user_id if the tool accepts it and LLM didn't provide one
        if tool_name in ["add_to_cart", "view_cart", "checkout_cart"]:
            if "user_id" not in tool_args:
                tool_args["user_id"] = user_id

        # Execute tool
        tool_function = tool_map[tool_name]
        tool_output = tool_function.invoke(tool_args)

        # Append result
        messages.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        )

        # Final pass: base LLM formats the response to the user
        final_response = llm.invoke(messages)
        print(f"\n🤖 Assistant:\n{final_response.content}\n")
        return final_response.content

    print(f"\n🤖 Assistant:\n{ai_response.content}\n")
    return ai_response.content