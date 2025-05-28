#!/usr/bin/env python3
"""
Debug Authentication Script
Checks the authentication environment to diagnose issues.
"""

import os
import json
from google.auth import default

def debug_authentication():
    print("🔍 Authentication Debug Information")
    print("=" * 50)
    
    # Check environment variables
    print("📋 Environment Variables:")
    print(f"   K_SERVICE: {os.getenv('K_SERVICE')}")
    print(f"   GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"   GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print()
    
    # Check api_key.json
    api_key_path = './api_key.json'
    print("📄 API Key File:")
    if os.path.exists(api_key_path):
        print(f"   ✓ api_key.json exists at: {api_key_path}")
        try:
            with open(api_key_path, 'r') as f:
                data = json.load(f)
                print(f"   ✓ Project ID in file: {data.get('project_id')}")
                print(f"   ✓ Service account: {data.get('client_email')}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    else:
        print(f"   ❌ api_key.json not found at: {api_key_path}")
    print()
    
    # Test default credentials
    print("🔐 Default Credentials Test:")
    try:
        credentials, project = default()
        print(f"   ✓ Default credentials work - Project: {project}")
        print(f"   ✓ Credentials type: {type(credentials)}")
        
        # Try to refresh the token
        credentials.refresh(None)
        print(f"   ✓ Token refresh successful")
        
    except Exception as e:
        print(f"   ❌ Default credentials failed: {e}")
    print()
    
    # Test setting GOOGLE_APPLICATION_CREDENTIALS manually
    print("🔧 Manual Credential Setup Test:")
    if os.path.exists(api_key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key_path
        print(f"   ✓ Set GOOGLE_APPLICATION_CREDENTIALS to: {api_key_path}")
        
        try:
            credentials, project = default()
            print(f"   ✓ Credentials work after manual setup - Project: {project}")
        except Exception as e:
            print(f"   ❌ Credentials still fail after manual setup: {e}")
    else:
        print(f"   ❌ Cannot test - api_key.json not found")

if __name__ == "__main__":
    debug_authentication() 