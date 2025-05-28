#!/usr/bin/env python3
"""
Test Vertex AI Authentication
Tests the exact authentication setup used in main.py
"""

import os
import json
from google.oauth2 import service_account
from google.auth import default
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

def test_vertex_auth():
    print("🧪 Testing Vertex AI Authentication")
    print("=" * 50)
    
    # Mimic the authentication setup from main.py
    try:
        # Try to load service account credentials explicitly first
        API_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.json")
        if os.path.exists(API_KEY_PATH) and not os.getenv("K_SERVICE"):
            # Local development - use service account file
            credentials = service_account.Credentials.from_service_account_file(API_KEY_PATH)
            with open(API_KEY_PATH, 'r') as f:
                key_data = json.load(f)
                project = key_data['project_id']
            print(f"✓ Loaded service account credentials - Project: {project}")
            print(f"✓ Service account: {key_data.get('client_email')}")
        else:
            # Cloud Run or fallback to default
            credentials, project = default()
            print(f"✓ Using default credentials - Project: {project}")
        
        print(f"✓ Credentials type: {type(credentials)}")
        
    except Exception as e:
        print(f"❌ Authentication setup failed: {e}")
        return False
    
    # Test VertexAI Embeddings
    print("\n🔤 Testing VertexAI Embeddings...")
    try:
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=project,
            location="us-central1",
            credentials=credentials
        )
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Embedding successful - dimension: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False
    
    # Test ChatVertexAI
    print("\n💬 Testing ChatVertexAI...")
    try:
        chat_model = ChatVertexAI(
            model_name="gemini-1.5-flash-001",
            project=project,
            location="us-central1",
            credentials=credentials,
            temperature=0.1
        )
        
        # Test a simple chat
        from langchain_core.messages import HumanMessage
        response = chat_model.invoke([HumanMessage(content="Say 'Hello, authentication test successful!'")])
        print(f"✓ Chat model successful - Response: {response.content}")
        
    except Exception as e:
        print(f"❌ Chat model test failed: {e}")
        return False
    
    print("\n🎉 All Vertex AI authentication tests passed!")
    return True

if __name__ == "__main__":
    success = test_vertex_auth()
    if not success:
        print("\n❌ Authentication tests failed. Check your credentials and project settings.")
        exit(1)
    else:
        print("\n✅ Authentication is working correctly for Vertex AI models.") 