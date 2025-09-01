"""
Live AD Group-Based Page Authentication
"""
import streamlit as st
import os
from utils.session_manager import EnhancedSessionManager
from login_page import show_login_page
from config import PAGE_ACCESS_CONFIG

def check_live_ad_page_access(page_name=None):
    """Complete version with role-based access control"""
    try:
        print("🔍 Starting page access check...")
        
        # Step 1: Check authentication
        if not EnhancedSessionManager.is_authenticated():
            print("User not authenticated, showing login page")
            show_login_page()
            st.stop()
        
        print("✅ Authentication check passed")
        
       
        # Step 3: Get user info
        user_info = EnhancedSessionManager.get_user_info()
        user_role = user_info.get('role', 'user')
        
        # Step 4: Check page access permissions
        allowed_roles = PAGE_ACCESS_CONFIG.get(page_name, ['admin'])  # Default: admin only
        
        # DEBUG: Print what's happening
        print(f"🔍 PAGE ACCESS DEBUG:")
        print(f"   Page: {page_name}")
        print(f"   User Role: {user_role}")
        print(f"   Allowed Roles: {allowed_roles}")
        print(f"   User in allowed roles: {user_role in allowed_roles}")
        
        # Step 5: Check if user has required role
        if user_role not in allowed_roles:
            print(f"   ❌ ACCESS DENIED for {user_role}")
            
            # Show access denied message
            st.error("🚫 Access Denied - Insufficient Permissions")
            

            st.info(f"**Your Access:** {user_info.get('display_name')} ({user_role.title()})")
            required_roles_text = " or ".join([role.title() for role in allowed_roles])
            st.info(f"**Required:** {required_roles_text}")
            

            st.markdown("### Actions")
            if st.button("🔄 Refresh Groups", help="Check for updated AD group memberships"):
                EnhancedSessionManager.force_group_refresh()
                st.rerun()
            
            if st.button("🚪 Logout", help="Login with different account"):
                EnhancedSessionManager.logout_user()
                st.rerun()
            
            st.markdown("---")
            st.markdown("💡 **Need Access?** Contact your IT administrator to be added to the appropriate AD groups.")
            
            st.stop()
        else:
            print(f"   ✅ ACCESS GRANTED for {user_role}")
        
        # Step 6: Access granted - show user info
        display_user_sidebar(user_info)
        
    except Exception as e:
        st.error(f"Authentication Error: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        show_login_page()
        st.stop()

  
def display_user_sidebar(user_info):
    """Display user information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Information")
        st.markdown(f"**Name:** {user_info.get('display_name', 'Unknown')}")
        st.markdown(f"**Role:** {user_info.get('role', 'user').title()}")
        st.markdown(f"**Auth:** {user_info.get('authenticated_via', 'Unknown')}")
        
        if user_info.get('ad_groups'):
            st.markdown(f"**AD Groups:** {', '.join(user_info.get('ad_groups', []))}")
        
        st.markdown(f"**Last Updated:** {user_info.get('last_updated', 'N/A')[:16]}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄", help="Refresh Groups"):
                EnhancedSessionManager.force_group_refresh()
                st.rerun()
        
        with col_b:
            if st.button("🚪", help="Logout"):
                EnhancedSessionManager.logout_user()
                st.rerun()


        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Information")
            st.markdown(f"**Name:** {user_info.get('display_name', 'Unknown')}")
            st.markdown(f"**Role:** {user_info.get('role', 'user').title()}")

def get_current_page_name():
    """Get current page name"""
    try:
        import inspect
        current_file = inspect.getfile(inspect.currentframe().f_back)
        return os.path.basename(current_file)
    except:
        return "unknown_page"