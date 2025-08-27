"""
Live AD Group Authentication Module
"""
import logging
import time
from datetime import datetime, timedelta
from ldap3 import Server, Connection, SIMPLE
from ldap3.core.exceptions import LDAPException
from config import (AD_SERVER, AD_PORT, AD_USE_SSL, AD_SERVICE_USER, AD_SERVICE_PASSWORD,
                   AD_SEARCH_BASES, AD_GROUPS_TO_ROLES, AD_CACHE_TTL_MINUTES)

logger = logging.getLogger(__name__)

class LiveADAuthenticator:
    def __init__(self):
        self.server = Server(AD_SERVER, port=AD_PORT, use_ssl=AD_USE_SSL)
        self._group_cache = {}  # Cache for group memberships
        self._cache_timestamps = {}  # When each cache entry was created
        
    def authenticate_user(self, username, password):
        """Authenticate user against AD and get live group memberships"""
        try:
            # Step 1: Authenticate user
            user_formats = [
                f"fsys\\{username}",        # Format that worked for users
                f"{username}@fsys.net"      # Alternative format
            ]
            
            authenticated = False
            for user_format in user_formats:
                try:
                    conn = Connection(
                        self.server,
                        user=user_format,
                        password=password,
                        authentication=SIMPLE,
                        auto_bind=True
                    )
                    
                    if conn.bind():
                        logger.info(f"User authentication successful: {username}")
                        conn.unbind()
                        authenticated = True
                        break
                        
                except Exception as e:
                    logger.debug(f"Auth failed for format {user_format}: {str(e)}")
                    continue
            
            if not authenticated:
                logger.warning(f"Authentication failed for user: {username}")
                return False, None
            
            # Step 2: Get live group memberships
            user_info = self._get_user_groups(username)
            if user_info:
                logger.info(f"Successfully retrieved groups for {username}: {user_info.get('groups', [])}")
                return True, user_info
            else:
                logger.error(f"Could not retrieve group information for {username}")
                return False, None
                
        except Exception as e:
            logger.error(f"Authentication error for {username}: {str(e)}")
            return False, None
    
    def _get_user_groups(self, username):
        """Get user's group memberships from AD with caching"""
        
        # Check cache first
        if self._is_cached(username):
            logger.debug(f"Using cached groups for {username}")
            return self._group_cache[username]
        
        try:
            # Connect with service account
            conn = Connection(
                self.server,
                user=AD_SERVICE_USER,
                password=AD_SERVICE_PASSWORD,
                authentication=SIMPLE,
                auto_bind=True
            )
            
            # Search in all configured bases
            user_entry = None
            for search_base in AD_SEARCH_BASES:
                logger.debug(f"Searching for {username} in {search_base}")
                
                conn.search(
                    search_base=search_base,
                    search_filter=f'(sAMAccountName={username})',
                    attributes=['displayName', 'mail', 'memberOf', 'sAMAccountName']
                )
                
                if conn.entries:
                    user_entry = conn.entries[0]
                    logger.debug(f"Found {username} in {search_base}")
                    break
            
            if not user_entry:
                logger.warning(f"User {username} not found in any search base")
                conn.unbind()
                return None
            
            # Extract group information
            groups = [str(group) for group in user_entry.memberOf.values] if user_entry.memberOf else []
            
            # Determine role based on AD groups
            role = self._determine_role_from_groups(groups)
            
            # Create user info
            user_info = {
                'username': username,
                'display_name': str(user_entry.displayName.value) if user_entry.displayName else username.title().replace('.', ' '),
                'email': str(user_entry.mail.value) if user_entry.mail else f'{username}@fsys.net',
                'role': role,
                'groups': groups,
                'ad_groups': [group for group in AD_GROUPS_TO_ROLES.keys() if any(group in g for g in groups)],
                'authenticated_via': 'Live AD',
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the result
            self._cache_user_info(username, user_info)
            
            conn.unbind()
            return user_info
            
        except Exception as e:
            logger.error(f"Error getting groups for {username}: {str(e)}")
            return None
    
    def _determine_role_from_groups(self, user_groups):
        """Determine user role from AD group memberships"""
        
        # Check for each role in priority order (admin > analyst > user)
        role_priority = ['admin', 'analyst', 'user']
        
        for role in role_priority:
            for ad_group, mapped_role in AD_GROUPS_TO_ROLES.items():
                if mapped_role == role:
                    # Check if user is in this AD group
                    if any(ad_group in group for group in user_groups):
                        logger.debug(f"User has role '{role}' from group '{ad_group}'")
                        return role
        
        # Default role if no groups match
        logger.warning(f"No matching AD groups found, assigning default 'user' role")
        return 'user'
    
    def _is_cached(self, username):
        """Check if user's group info is cached and still valid"""
        if username not in self._group_cache:
            return False
        
        cache_time = self._cache_timestamps.get(username)
        if not cache_time:
            return False
        
        # Check if cache is still valid
        cache_age = datetime.now() - cache_time
        return cache_age < timedelta(minutes=AD_CACHE_TTL_MINUTES)
    
    def _cache_user_info(self, username, user_info):
        """Cache user information with timestamp"""
        self._group_cache[username] = user_info
        self._cache_timestamps[username] = datetime.now()
        
        # Clean old cache entries if needed
        if len(self._group_cache) > 100:  # Prevent memory issues
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove old cache entries"""
        now = datetime.now()
        expired_users = []
        
        for username, timestamp in self._cache_timestamps.items():
            if now - timestamp > timedelta(minutes=AD_CACHE_TTL_MINUTES):
                expired_users.append(username)
        
        for username in expired_users:
            if username in self._group_cache:
                del self._group_cache[username]
            if username in self._cache_timestamps:
                del self._cache_timestamps[username]
        
        logger.debug(f"Cleaned {len(expired_users)} expired cache entries")
    
    def refresh_user_groups(self, username):
        """Force refresh of user's group memberships (bypass cache)"""
        if username in self._group_cache:
            del self._group_cache[username]
        if username in self._cache_timestamps:
            del self._cache_timestamps[username]
        
        return self._get_user_groups(username)
    
    def get_cached_user_info(self, username):
        """Get cached user info if available"""
        if self._is_cached(username):
            return self._group_cache[username]
        return None