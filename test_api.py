#!/usr/bin/env python3

import requests
import json

def test_api():
    """Test the API endpoint directly"""
    
    url = "http://localhost:5000/chat_rag"
    
    test_data = {
        "query": "How to identify Dental Claim?",
        "model": "gemini-2.5-pro-preview-05-06"
    }
    
    try:
        print("🧪 Testing API endpoint...")
        print(f"📝 Query: {test_data['query']}")
        print("-" * 50)
        
        response = requests.post(url, json=test_data)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS - Response received:")
            print(f"  Answer length: {len(data.get('answer', ''))}")
            print(f"  Citations count: {len(data.get('citations', []))}")
            print(f"  Confidence: {data.get('confidence_score', 0)}")
            
            if data.get('citations'):
                print("\n📚 Citations preview:")
                for i, citation in enumerate(data['citations'][:3]):
                    print(f"  [{i+1}] {citation.get('formatted_source', 'No source')}")
        else:
            print(f"\n❌ ERROR - Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server. Is it running on localhost:5000?")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_api() 