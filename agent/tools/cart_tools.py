from langchain_core.tools import tool
from agent.config.supabase import supabase
from agent.services.payment import test_create_payment_link

# Import your Safepay function (adjust the path to wherever you saved it)
# from agent.services.payment import create_safepay_checkout 

@tool
def add_to_cart(
    product_title_or_id: str, quantity: int = 1, user_id: str = "guest_user"
) -> str:
    """Use this tool when the user wants to add an item to their shopping cart.
    Accepts product name/id and quantity.
    """
    prod_res = (
        supabase.table("products")
      .select("id, title, price, stock_quantity")
      .ilike("title", f"%{product_title_or_id}%")
      .limit(1)
      .execute()
    )

    if not prod_res.data:
        return f"Could not find product '{product_title_or_id}' to add to your cart."

    product = prod_res.data[0]

    if product["stock_quantity"] < quantity:
        return (
        f"Cannot add {quantity} units. Only {product['stock_quantity']} units"
        f" of '{product['title']}' available."
    )

    cart_res = (
        supabase.table("cart_items")
        .select("id, quantity")
        .eq("user_id", user_id)
        .eq("product_id", product["id"])
        .execute()
    )

    if cart_res.data:
        new_qty = cart_res.data[0]["quantity"] + quantity
        supabase.table("cart_items").update({"quantity": new_qty}).eq(
            "id", cart_res.data[0]["id"]
        ).execute()
    else:
        supabase.table("cart_items").insert({
            "user_id": user_id,
            "product_id": product["id"],
            "quantity": quantity,
        }).execute()

    return f"Added {quantity}x '{product['title']}' (PKR {product['price']} each) to your cart."


@tool
def view_cart(user_id: str = "guest_user") -> str:
    """Use this tool when the user asks to view, check, or summarize their cart contents."""
    res = (
        supabase.table("cart_items")
        .select("quantity, products(title, price)")
        .eq("user_id", user_id)
        .execute()
    )

    if not res.data:
        return "Your shopping cart is currently empty."

    total = 0.0
    lines = ["🛒 Your Shopping Cart:"]
    for item in res.data:
        prod = item["products"]
        subtotal = item["quantity"] * float(prod["price"])
        total += subtotal
        lines.append(
            f"- {prod['title']} x{item['quantity']} = PKR {subtotal:.2f} (PKR {prod['price']} ea)"
        )

    lines.append(f"\nTotal: PKR {total:.2f}")
    return "\n".join(lines)


@tool
def confirm_order(shipping_address: str, user_id: str = "guest_user") -> str:
    """Use this tool to finalize the order, clear the cart, and generate a payment link.
    Requires a valid delivery/shipping address.
    """
    # 1. Fetch current cart from Supabase (Highly Secure - ignores LLM hallucinations)
    cart_res = (
        supabase.table("cart_items")
        .select("id, product_id, quantity, products(id, title, price, stock_quantity)")
        .eq("user_id", user_id)
        .execute()
    )

    items = cart_res.data or []
    if not items:
        return "Your cart is empty. Please add items before confirming your order."

    # 2. Verify inventory and securely calculate the exact total
    total_price = 0.0
    for item in items:
        p = item["products"]
        if p["stock_quantity"] < item["quantity"]:
            return f"Checkout failed: '{p['title']}' has insufficient stock ({p['stock_quantity']} left)."
        total_price += item["quantity"] * float(p["price"])

    # 3. Create the Order in Supabase (Awaiting Payment)
    order_res = (
        supabase.table("orders")
        .insert({
            "user_id": user_id,
            "total_amount": total_price,
            "shipping_address": shipping_address,
            "status": "pending", 
        })
        .execute()
    )
    
    order_id = order_res.data[0]["id"]

    # 4. Create Order Items & Decrement Inventory
    for item in items:
        p = item["products"]
        supabase.table("order_items").insert({
            "order_id": order_id,
            "product_id": p["id"],
            "quantity": item["quantity"],
            "unit_price": p["price"],
        }).execute()

        supabase.table("products").update(
            {"stock_quantity": p["stock_quantity"] - item["quantity"]}
        ).eq("id", p["id"]).execute()

    # 5. Clear the User's Cart
    supabase.table("cart_items").delete().eq("user_id", user_id).execute()

    # 6. Generate the Safepay Checkout Link
    try:
        # Assuming you imported test_create_payment_link at the top of the file
        checkout_url = test_create_payment_link(order_id, total_price)
        
        return (
            f"Order placed successfully! Total: PKR {total_price:.2f}. "
            f"Please complete your secure payment here: {checkout_url}"
        )
    except Exception as e:
        print(f"Payment generation failed: {e}")
        return "Your order was reserved, but the payment gateway is currently unavailable. Please try again later."