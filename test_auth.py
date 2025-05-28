#!/usr/bin/env python3
"""
Authentication Test Script
Tests that GCP authentication is working correctly for both local and Cloud Run environments.
"""

import os
import json
from pathlib import Path

def test_authentication():
    """Test the authentication setup"""
    print("🔍 Testing AET-RAG Authentication Setup")
    print("=" * 50)
    
    # Check environment
    is_cloud_run = bool(os.getenv("K_SERVICE"))
    print(f"Environment: {'Cloud Run' if is_cloud_run else 'Local Development'}")
    
    if is_cloud_run:
        print("✓ Running in Cloud Run - using Workload Identity Federation")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if project_id:
            print(f"✓ Project ID from environment: {project_id}")
        else:
            print("❌ GOOGLE_CLOUD_PROJECT not set")
            return False
    else:
        print("🏠 Running locally - checking credential options...")
        
        # Check for api_key.json
        api_key_path = Path("api_key.json")
        if api_key_path.exists():
            print(f"✓ Found api_key.json at: {api_key_path.absolute()}")
            try:
                with open(api_key_path, 'r') as f:
                    key_data = json.load(f)
                    project_id = key_data.get('project_id')
                    if project_id:
                        print(f"✓ Project ID from key file: {project_id}")
                    else:
                        print("⚠ No project_id in key file")
                    
                    client_email = key_data.get('client_email')
                    if client_email:
                        print(f"✓ Service account: {client_email}")
                    
            except Exception as e:
                print(f"❌ Error reading api_key.json: {e}")
                return False
        else:
            print("⚠ api_key.json not found")
            print("💡 Checking for Application Default Credentials...")
            
            # Check for ADC
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            if adc_path.exists():
                print(f"✓ Found Application Default Credentials at: {adc_path}")
            else:
                print("❌ No Application Default Credentials found")
                print("💡 Run: gcloud auth application-default login")
                return False
    
    # Test GCP connection
    print("\n🧪 Testing GCP Connection...")
    try:
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
        print(f"✓ Successfully loaded credentials")
        print(f"✓ Project: {project}")
        
        # Test token refresh
        request = Request()
        credentials.refresh(request)
        print("✓ Successfully refreshed token")
        
        return True
        
    except Exception as e:
        print(f"❌ GCP authentication failed: {e}")
        return False

def test_vertex_ai():
    """Test Vertex AI connection"""
    print("\n🤖 Testing Vertex AI Connection...")
    try:
        from langchain_google_vertexai import VertexAIEmbeddings
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "aethrag2")
        location = os.getenv("GCP_LOCATION", "us-east1")
        
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=project_id,
            location=location
        )
        
        # Test with a simple embedding
        test_text = "Hello, this is a test."
        result = embeddings.embed_query(test_text)
        
        print(f"✓ Successfully created embedding (dimension: {len(result)})")
        return True
        
    except Exception as e:
        print(f"❌ Vertex AI test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 AET-RAG Authentication Test")
    print("=" * 50)
    
    # Test basic authentication
    auth_ok = test_authentication()
    
    if auth_ok:
        # Test Vertex AI if basic auth works
        vertex_ok = test_vertex_ai()
        
        if vertex_ok:
            print("\n🎉 All tests passed! Authentication is working correctly.")
            print("\n📝 Next steps:")
            print("1. Run: python create_chroma_db.py (if not done already)")
            print("2. Run: python main.py")
            print("3. Open: http://localhost:8080")
        else:
            print("\n⚠ Basic authentication works, but Vertex AI connection failed.")
            print("Check your GCP project settings and API enablement.")
    else:
        print("\n❌ Authentication test failed.")
        print("\n💡 Setup instructions:")
        print("1. See LOCAL_SETUP.md for detailed setup instructions")
        print("2. Create api_key.json or run: gcloud auth application-default login")
        print("3. Ensure your GCP project has Vertex AI API enabled")

if __name__ == "__main__":
    main() 