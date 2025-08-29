# """
# Session Management for AD Authentication
# """
# import streamlit as st
# from datetime import datetime, timedelta

# class SessionManager:
#     @staticmethod
#     def login_user(user_info):
#         st.session_state.authenticated = True
#         st.session_state.user_info = user_info
#         st.session_state.login_time = datetime.now()
        
#     @staticmethod
#     def logout_user():
#         for key in ['authenticated', 'user_info', 'login_time']:
#             if key in st.session_state:
#                 del st.session_state[key]
    
#     @staticmethod
#     def is_authenticated():
#         return st.session_state.get('authenticated', False)
    
#     @staticmethod
#     def get_user_info():
#         return st.session_state.get('user_info', {})
    
#     @staticmethod
#     def has_role(required_role):
#         if not SessionManager.is_authenticated():
#             return False
            
#         user_role = SessionManager.get_user_info().get('role', 'user')
        
#         role_hierarchy = ['user', 'analyst', 'admin']
#         user_level = role_hierarchy.index(user_role) if user_role in role_hierarchy else 0
#         required_level = role_hierarchy.index(required_role) if required_role in role_hierarchy else 0
        
#         return user_level >= required_level


"""
Enhanced Session Management with Live Group Refresh
"""
import streamlit as st
from datetime import datetime, timedelta

class EnhancedSessionManager:
    @staticmethod
    def login_user(user_info):
        """Store user info in session with enhanced data"""
        st.session_state.authenticated = True
        st.session_state.user_info = user_info
        st.session_state.login_time = datetime.now()
        st.session_state.last_group_check = datetime.now()
    
        # Debug: verify session is set
        print(f"✅ Session set: {st.session_state.authenticated}")
        print(f"✅ User info stored: {user_info.get('username')}")
        
    @staticmethod
    def logout_user():
        """Clear all session data"""
        session_keys = ['authenticated', 'user_info', 'login_time', 'last_group_check']
        for key in session_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def is_authenticated():
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def get_user_info():
        """Get current user info"""
        return st.session_state.get('user_info', {})
    
    @staticmethod
    def check_session_timeout(timeout_hours=8):
        """Check if session has timed out"""
        if not EnhancedSessionManager.is_authenticated():
            return True
            
        login_time = st.session_state.get('login_time')
        if login_time:
            if datetime.now() - login_time > timedelta(hours=timeout_hours):
                EnhancedSessionManager.logout_user()
                return True
            else:
                return False  # Session is valid
        else:
            return True  # No login time means not logged in
    
    @staticmethod
    def has_role(required_role):
        """Check if user has required role with live group refresh"""
        if not EnhancedSessionManager.is_authenticated():
            return False
        
        # Check if we need to refresh groups (every 5 minutes)
        last_check = st.session_state.get('last_group_check')
        if last_check and datetime.now() - last_check > timedelta(minutes=5):
            EnhancedSessionManager._refresh_user_groups()
        
        user_role = EnhancedSessionManager.get_user_info().get('role', 'user')
        
        role_hierarchy = ['user', 'analyst', 'admin']
        user_level = role_hierarchy.index(user_role) if user_role in role_hierarchy else 0
        required_level = role_hierarchy.index(required_role) if required_role in role_hierarchy else 0
        
        return user_level >= required_level
    
    @staticmethod
    def _refresh_user_groups():
        """Refresh user's group memberships from AD"""
        try:
            from utils.ad_auth import LiveADAuthenticator
            
            user_info = st.session_state.get('user_info', {})
            username = user_info.get('username')
            
            if username:
                auth = LiveADAuthenticator()
                fresh_user_info = auth.refresh_user_groups(username)
                
                if fresh_user_info:
                    # Update session with fresh group info
                    st.session_state.user_info = fresh_user_info
                    st.session_state.last_group_check = datetime.now()
                    
                    # Check if role changed
                    old_role = user_info.get('role')
                    new_role = fresh_user_info.get('role')
                    
                    if old_role != new_role:
                        st.success(f"Your access level has been updated: {old_role} → {new_role}")
                        st.rerun()
                        
        except Exception as e:
            # Don't break the app if group refresh fails
            import logging
            logging.error(f"Failed to refresh user groups: {str(e)}")
    
    @staticmethod
    def force_group_refresh():
        """Manually force a group refresh (for admin tools)"""
        EnhancedSessionManager._refresh_user_groups()