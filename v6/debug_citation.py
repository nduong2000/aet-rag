#!/usr/bin/env python3

import json
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import AetnaDataScienceRAGSystem

def test_citations():
    print("🧪 Testing citation generation...")
    
    # Initialize the system
    rag_system = AetnaDataScienceRAGSystem()
    
    # Test query
    query = "How to identify Dental Claim?"
    
    print(f"📝 Query: {query}")
    print("-" * 50)
    
    # Process query
    result = rag_system.process_query(query)
    
    # Print the raw result for debugging
    print("📊 Raw API Response:")
    print(json.dumps(result, indent=2, default=str))
    
    print("\n" + "=" * 50)
    print("🔍 Citation Analysis:")
    
    if "citations" in result:
        citations = result["citations"]
        print(f"📚 Number of citations: {len(citations)}")
        
        for i, citation in enumerate(citations[:3]):  # Show first 3
            print(f"\n[{i+1}] Citation structure:")
            for key, value in citation.items():
                print(f"  {key}: {value}")
    else:
        print("❌ No citations found in response")
    
    print("\n" + "=" * 50)
    print("🎯 Frontend Display Preview:")
    
    if "citations" in result and result["citations"]:
        for citation in result["citations"][:3]:
            source = citation.get("formatted_source", citation.get("source_file", "Unknown"))
            doc_type = citation.get("document_type", "Unknown")
            index = citation.get("index", "?")
            print(f"[{index}] {source} ({doc_type})")
    else:
        print("No citations to display")

if __name__ == "__main__":
    test_citations() 