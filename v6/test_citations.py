#!/usr/bin/env python3
"""
Test script to verify the enhanced citation system
"""

import os
import sys
import json
from main import AetnaDataScienceRAGSystem

def test_citation_system():
    """Test the enhanced citation system with real queries"""
    print("🔍 Testing Enhanced Citation System")
    print("=" * 50)
    
    # Set up GCP authentication 
    api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.json")
    if os.path.exists(api_key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key_path
    
    try:
        # Initialize the RAG system
        print("1. Initializing RAG system...")
        rag_system = AetnaDataScienceRAGSystem()
        
        if not rag_system.collection:
            print("❌ ChromaDB collection not available")
            return False
        
        # Test queries specifically about identifiable content
        test_queries = [
            "How to identify Dental Claim?",
            "What is field 15 in the Universal file?",
            "Explain External Stop Loss reporting requirements",
            "What are the capitation payment file specifications?",
            "Where can I find provider identification fields?"
        ]
        
        print("\n2. Testing queries with citation tracking...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            
            try:
                result = rag_system.process_query(query)
                
                # Check if citations are present
                citations = result.get("citations", [])
                
                print(f"✓ Answer generated (confidence: {result.get('confidence_score', 0):.2f})")
                print(f"✓ Citations found: {len(citations)}")
                
                if citations:
                    print("\n📚 Citation Details:")
                    for cite in citations[:3]:  # Show first 3 citations
                        print(f"  [{cite['index']}] {cite['formatted_source']}")
                        print(f"      Type: {cite['document_type']}")
                        if cite.get('section_title'):
                            print(f"      Section: {cite['section_title']}")
                        if cite.get('contains_field_definitions'):
                            print(f"      Contains field definitions: {cite['field_numbers'][:3]}")
                        print()
                
                # Check if answer contains source references
                answer = result.get("answer", "")
                has_sources_section = "**Sources:**" in answer
                print(f"✓ Sources section in answer: {has_sources_section}")
                
                # Show part of the answer to verify citations are included
                answer_preview = answer[:500] + "..." if len(answer) > 500 else answer
                print(f"\n📝 Answer preview:\n{answer_preview}")
                
                if not citations:
                    print("⚠️  No citations found - this might indicate an issue")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
                continue
        
        print("\n3. Testing specific document retrieval...")
        
        # Test document metadata directly
        if rag_system.collection:
            try:
                # Query for documents and check metadata
                query_embedding = rag_system.embeddings.embed_query("dental claim")
                results = rag_system.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                print("✓ Direct document query successful")
                
                if results["metadatas"] and results["metadatas"][0]:
                    print("\n📄 Sample document metadata:")
                    for i, metadata in enumerate(results["metadatas"][0][:2]):
                        print(f"  Document {i+1}:")
                        print(f"    Source file: {metadata.get('source_file', 'Unknown')}")
                        print(f"    Page/Position: {metadata.get('page', metadata.get('estimated_page', 'Unknown'))}")
                        print(f"    Document type: {metadata.get('primary_doc_type', 'Unknown')}")
                        print(f"    Section: {metadata.get('section_title', 'N/A')}")
                        print(f"    Field definitions: {metadata.get('contains_field_definitions', False)}")
                        print()
                
            except Exception as e:
                print(f"⚠️  Direct document query failed: {e}")
        
        print("\n🎉 Citation system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_citation_format():
    """Test citation formatting specifically"""
    print("\n🔧 Testing Citation Formatting")
    print("=" * 40)
    
    # Mock document data for testing
    mock_documents = [
        {
            "page_content": "Test content about dental claims",
            "metadata": {
                "source_file": "Universal_File_Layout.pdf",
                "estimated_page": 15,
                "section_title": "Dental Claims Processing",
                "primary_doc_type": "universal_medical_file",
                "contains_field_definitions": True,
                "field_numbers": ["15", "16", "17"],
                "chunk_index": 3
            }
        },
        {
            "page_content": "Test content about stop loss",
            "metadata": {
                "source_file": "External_Stop_Loss_Report.xlsx",
                "page": 5,
                "primary_doc_type": "external_stop_loss",
                "contains_field_definitions": False,
                "chunk_index": 1
            }
        }
    ]
    
    try:
        rag_system = AetnaDataScienceRAGSystem()
        
        # Test citation extraction
        mock_state = {"filtered_documents": mock_documents}
        citations = rag_system._extract_citations(mock_state)
        
        print(f"✓ Extracted {len(citations)} citations")
        
        for cite in citations:
            print(f"  [{cite['index']}] {cite['formatted_source']}")
            print(f"      Type: {cite['document_type']}")
            if cite.get('section_title'):
                print(f"      Section: {cite['section_title']}")
        
        # Test citation formatting
        citation_text = rag_system._format_citations_text(citations)
        print(f"\n📝 Formatted citation text:\n{citation_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Citation formatting test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_citation_system()
    format_success = test_citation_format()
    
    if not (success and format_success):
        sys.exit(1)
    
    print("\n✅ All citation tests passed!") 