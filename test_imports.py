# test_imports.py
try:
    from utils.ad_auth import LiveADAuthenticator
    print("✅ LiveADAuthenticator imported successfully")
except Exception as e:
    print(f"❌ LiveADAuthenticator import failed: {str(e)}")

try:
    from utils.session_manager import EnhancedSessionManager
    print("✅ EnhancedSessionManager imported successfully")  
except Exception as e:
    print(f"❌ EnhancedSessionManager import failed: {str(e)}")

try:
    from utils.page_auth import check_live_ad_page_access
    print("✅ page_auth imported successfully")
except Exception as e:
    print(f"❌ page_auth import failed: {str(e)}")

print("Import test complete")