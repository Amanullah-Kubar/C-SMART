import os
import httpx
from dotenv import load_dotenv
load_dotenv()
SAFEPAY_API_KEY = os.getenv("SAFEPAY_API_KEY")  # Paste your sandbox secret key

BASE_URL = "https://sandbox.api.getsafepay.com"
CHECKOUT_URL = "https://sandbox.api.getsafepay.com/checkout/pay"

def test_create_payment_link(order_id: str, amount_pkr: float):
    print(f"--- Initializing Safepay Order #{order_id} for PKR {amount_pkr} ---")
    
    url = f"{BASE_URL}/order/v1/init"
    headers = {
        "X-SFPY-MERCHANT-SECRET": SAFEPAY_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "client": SAFEPAY_API_KEY,
        "amount": float(amount_pkr),
        "currency": "PKR",
        "environment": "sandbox", # test enviroment
        "metadata":{
            "order_id": order_id,
        }
    }

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print("Raw API Response:", data)

        if response.status_code == 200 and "data" in data:
            tracker_token = data["data"]["token"]
            # if there is the sandbox enviroment
            checkout_link = f"{CHECKOUT_URL}?beacon={tracker_token}&env=sandbox" 
            
            print("\n" + "="*60)
            print(" PAYMENT LINK GENERATED SUCCESSFULLY:")
            print(checkout_link)
            print("="*60)
            print("\nNext step: Open this link in your browser to complete a test payment.")
            return checkout_link
        else:
            print("Failed to initialize transaction:", data.get("message"))
            return None

    except Exception as e:
        print(f"Connection error: {e}")
        return None