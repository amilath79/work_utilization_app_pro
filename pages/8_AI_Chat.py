"""
AI Chat page for workforce prediction queries
"""
import streamlit as st
import requests
import json
from datetime import datetime
import uuid
from utils.page_auth import check_live_ad_page_access
# Configure page
st.set_page_config(
    page_title="AI Workforce Chat",
    page_icon="🤖",
    layout="wide"
)


check_live_ad_page_access('')
def send_chat_message(message, webhook_url, session_id=None):
    """Send message to AI chat webhook with session management"""
    try:
        # n8n chat trigger expects this specific format
        payload = {
            "action": "sendMessage",
            "sessionId": session_id,
            "chatInput": message
        }
        
        response = requests.post(
            webhook_url, 
            json=payload, 
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract the actual message from n8n response
            if isinstance(result, dict):
                return result.get("output", result.get("response", result.get("text", str(result))))
            return result
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - please try again"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - check if n8n is running"}
    except Exception as e:
        return {"error": str(e)}

def main():
    st.header("🤖 AI Workforce Prediction Chat")
    st.caption("Ask questions about your workforce data and predictions")
    
    # Webhook URL
    webhook_url = "http://192.168.1.42:5678/webhook/afcc6d8a-8d52-4a96-b6d3-3554c35efc68/chat"
    
    # Initialize session ID for memory persistence
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display connection status
    with st.sidebar:
        st.subheader("Connection Status")
        try:
            # Test connection
            test_response = requests.get("http://localhost:5678/healthz", timeout=5)
            if test_response.status_code == 200:
                st.success("✅ n8n Connected")
            else:
                st.error("❌ n8n Not Responding")
        except:
            st.error("❌ n8n Not Connected")
        
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        if st.button("🔄 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.rerun()
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What workforce data do we have?"):
            st.session_state.pending_message = "What workforce data do we have in our database?"
            st.rerun()
        
        if st.button("Show me punch code 206 trends"):
            st.session_state.pending_message = "Show me trends for punch code 206"
            st.rerun()
    
    with col2:
        if st.button("Predict next week's workforce"):
            st.session_state.pending_message = "Can you predict next week's workforce requirements?"
            st.rerun()
            
        if st.button("Compare work types"):
            st.session_state.pending_message = "Compare different work types in our data"
            st.rerun()
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Create chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    
    # Handle pending message from sample buttons
    if hasattr(st.session_state, 'pending_message'):
        user_input = st.session_state.pending_message
        del st.session_state.pending_message
    else:
        user_input = st.chat_input("Ask about workforce predictions, data analysis, or trends...")
    
    # Process user input
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
            with st.spinner("🤔 Analyzing your question..."):
                response = send_chat_message(
                    user_input, 
                    webhook_url, 
                    st.session_state.session_id
                )
                
                if "error" in response:
                    ai_response = f"❌ {response['error']}\n\nPlease check:\n- Is n8n running?\n- Is the workflow active?\n- Are all nodes properly connected?"
                    st.error(ai_response)
                else:
                    # Extract response from different possible formats
                    if isinstance(response, dict):
                        ai_response = (
                            response.get("output", "") or 
                            response.get("response", "") or 
                            response.get("text", "") or
                            str(response)
                        )
                    else:
                        ai_response = str(response)
                    
                    if ai_response.strip():
                        st.write(ai_response)
                    else:
                        st.warning("⚠️ Received empty response from AI")
                        ai_response = "No response received"
        
        # Add AI response to history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now()
        })
        
        # Rerun to show the new messages
        st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat"):
            if st.session_state.chat_messages:
                chat_export = []
                for msg in st.session_state.chat_messages:
                    chat_export.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"].isoformat()
                    })
                
                st.download_button(
                    label="Download Chat History",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"workforce_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    # Footer info
    st.markdown("---")
    st.caption(f"💡 **Tip:** This AI can help you analyze workforce data, make predictions, and answer questions about your database.")

if __name__ == "__main__":
    main()