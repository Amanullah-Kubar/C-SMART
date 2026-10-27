from langchain_core.messages import HumanMessage, AIMessage
from agent.config.supabase import supabase

def get_recent_history(user_id: str, limit: int = 6):
    """Database se aakhri N messages utha kar LangChain format mein convert karta hai."""
    try:
        # Latest messages descending order mein lein
        response = (
            supabase.table("chat_history")
            .select("role, content")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        
        raw_msgs = response.data or []
        # Choonke query DESC thi, unhe reverse karke chronological (purani se nayi) karein
        raw_msgs.reverse()
        
        chat_messages = []
        for msg in raw_msgs:
            if msg["role"] == "user":
                chat_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_messages.append(AIMessage(content=msg["content"]))
                
        return chat_messages
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []

def save_message(user_id: str, role: str, content: str):
    """User aur bot ke messages ko persist karta hai."""
    try:
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        print(f"Error saving message to history: {e}")