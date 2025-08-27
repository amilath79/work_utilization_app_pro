"""
Login page with Live AD Integration
"""
import streamlit as st
from utils.ad_auth import LiveADAuthenticator
from utils.session_manager import EnhancedSessionManager
from config import AD_ENABLED, FALLBACK_AUTH_ENABLED
import logging

logger = logging.getLogger(__name__)

def show_login_page():
    """Display login form with live AD integration"""
    st.title("🔐 Workforce Prediction - Live AD Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### Enterprise Login")
            st.markdown("*Your access is controlled by IT through Active Directory groups*")
            
            username = st.text_input("Username", placeholder="Enter your AD username")
            password = st.text_input("Password", type="password", placeholder="Enter your AD password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_button = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")
            with col_b:
                refresh_button = st.form_submit_button("🔄 Refresh Groups", use_container_width=True)
            
            if login_button:
                if username and password:
                    authenticate_user(username, password)
                else:
                    st.error("Please enter both username and password")
            
            if refresh_button:
                if EnhancedSessionManager.is_authenticated():
                    with st.spinner("Refreshing your group memberships..."):
                        EnhancedSessionManager.force_group_refresh()
                    st.success("Groups refreshed!")
                else:
                    st.error("Please login first")

def authenticate_user(username, password):
    """Authenticate user with Live AD"""
    with st.spinner("Authenticating with Active Directory..."):
        
        if AD_ENABLED:
            # Try Live AD authentication
            auth = LiveADAuthenticator()
            success, user_info = auth.authenticate_user(username, password)
            
            if success and user_info:
                EnhancedSessionManager.login_user(user_info)
                
                # Show success message with group info
                st.success(f"Welcome {user_info.get('display_name', username)}!")
                
                with st.expander("ℹ️ Your Access Details"):
                    st.write(f"**Role:** {user_info.get('role', 'user').title()}")
                    st.write(f"**AD Groups:** {', '.join(user_info.get('ad_groups', []))}")
                    st.write(f"**Email:** {user_info.get('email', 'N/A')}")
                    st.write(f"**Last Updated:** {user_info.get('last_updated', 'N/A')}")
                
                # Auto-refresh after 2 seconds
                time.sleep(2)
                st.rerun()
                
            else:
                # Try fallback if enabled
                if FALLBACK_AUTH_ENABLED and fallback_authenticate(username, password):
                    user_info = create_fallback_user_info(username)
                    EnhancedSessionManager.login_user(user_info)
                    st.warning(f"Welcome {username}! (Using fallback authentication)")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or insufficient permissions")
                    st.info("💡 Contact IT if you believe you should have access")
        else:
            # AD disabled - use fallback only
            if fallback_authenticate(username, password):
                user_info = create_fallback_user_info(username)
                EnhancedSessionManager.login_user(user_info)
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

def fallback_authenticate(username, password):
    """Simple fallback authentication for testing"""
    # Simple test - replace with your logic
    return password == "test123"

def create_fallback_user_info(username):
    """Create fallback user info (for testing only)"""
    fallback_roles = {
        'amila.g': 'admin',
        'amila': 'admin',
        'mattias': 'analyst',
        'david': 'user'
    }
    
    return {
        'username': username,
        'display_name': username.title().replace('.', ' '),
        'email': f'{username}@fsys.net',
        'role': fallback_roles.get(username.lower(), 'user'),
        'groups': [],
        'ad_groups': [],
        'authenticated_via': 'Fallback',
        'last_updated': datetime.now().isoformat()
    }
