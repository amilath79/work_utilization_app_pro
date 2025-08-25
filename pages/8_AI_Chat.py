"""
AI Chat page for workforce prediction queries
"""
import streamlit as st
import requests
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Workforce Chat",
    page_icon="🤖",
    layout="wide"
)

def send_chat_message(message, webhook_url):
    """Send message to AI chat webhook"""
    try:
        payload = {"message": message}
        response = requests.post(webhook_url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    st.header("🤖 AI Workforce Prediction Chat")
    
    # Webhook URL
    webhook_url = "http://192.168.1.42:5678/webhook/afcc6d8a-8d52-4a96-b6d3-3554c35efc68/chat"
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Chat interface
    st.subheader("Ask questions about workforce predictions")
    
    # Display chat history
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your workforce prediction question...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Send to AI and get response
        with st.chat_message("assistant"):
            with st.spinner("Getting AI response..."):
                response = send_chat_message(user_input, webhook_url)
                
                if "error" in response:
                    ai_response = f"Error: {response['error']}"
                    st.error(ai_response)
                else:
                    ai_response = response.get("response", "No response received")
                    st.write(ai_response)
        
        # Add AI response to history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now()
        })
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()

if __name__ == "__main__":
    main()