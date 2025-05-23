#!/usr/bin/env python3
"""
Test script to verify the embedding dimension fix
"""

import os
import sys
from main import AetnaDataScienceRAGSystem
from langchain_google_vertexai import VertexAIEmbeddings

def test_embedding_dimensions():
    """Test that embeddings have correct dimensions"""
    print("🧪 Testing Embedding Dimensions")
    print("=" * 40)
    
    # Set up GCP authentication 
    api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.json")
    if os.path.exists(api_key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key_path
    
    try:
        # Test VertexAI embedding model directly
        print("1. Testing VertexAI Embedding Model...")
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project="aethrag2",
            location="us-central1"
        )
        
        test_query = "How to identify dental claims?"
        query_embedding = embeddings.embed_query(test_query)
        
        print(f"   Query: '{test_query}'")
        print(f"   Embedding dimensions: {len(query_embedding)}")
        
        if len(query_embedding) == 768:
            print("   ✅ Correct dimensions (768)")
        else:
            print(f"   ❌ Wrong dimensions (expected 768, got {len(query_embedding)})")
            return False
        
        # Test batch embeddings
        test_docs = ["Dental claim processing", "Medical claim processing"]
        doc_embeddings = embeddings.embed_documents(test_docs)
        
        print(f"   Document embeddings: {len(doc_embeddings)} docs")
        print(f"   Each embedding dimensions: {len(doc_embeddings[0])}")
        
        if all(len(emb) == 768 for emb in doc_embeddings):
            print("   ✅ All document embeddings have correct dimensions")
        else:
            print("   ❌ Document embeddings have wrong dimensions")
            return False
        
        print("\n2. Testing RAG System Integration...")
        
        # Test the RAG system
        try:
            rag_system = AetnaDataScienceRAGSystem()
            
            if rag_system.embeddings and rag_system.collection:
                print("   ✅ RAG system initialized successfully")
                
                # Test a simple query
                result = rag_system.process_query("test query")
                print(f"   ✅ Query processing works (confidence: {result.get('confidence_score', 0):.2f})")
                
            else:
                print("   ⚠️ RAG system missing components")
                print(f"      Embeddings: {'✓' if rag_system.embeddings else '✗'}")
                print(f"      Collection: {'✓' if rag_system.collection else '✗'}")
                
        except Exception as e:
            print(f"   ⚠️ RAG system test failed: {e}")
        
        print("\n🎉 Embedding dimension test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_dimensions()
    if not success:
        sys.exit(1) 