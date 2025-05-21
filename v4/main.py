# Enhanced RAG system with Advanced LangGraph workflow
# Optimized for numbered definitions, embedded tables, and medical data dictionary

import os
from dotenv import load_dotenv 
from flask import Flask, request, jsonify, render_template 
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
import chromadb
import sys 
import re 
from typing import TypedDict, List, Dict, Optional, Literal, Union, Any
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.graph import CompiledGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph not available. Using fallback implementation.")
    LANGGRAPH_AVAILABLE = False
    # Create dummy classes for fallback
    class StateGraph:
        def __init__(self, state_type): pass
        def add_node(self, name, func): pass
        def add_edge(self, source, target): pass
        def add_conditional_edges(self, source, condition, mapping): pass
        def set_entry_point(self, node): pass
        def compile(self): return FallbackGraph()
    
    class FallbackGraph:
        def invoke(self, state): return fallback_rag_pipeline(state)
    
    START = "START"
    END = "END"
    CompiledGraph = FallbackGraph

import logging

load_dotenv() 

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db_data"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL_NAME = "text-embedding-005"
CHAT_MODEL_NAME = "gemini-2.5-pro-preview-05-06"  # Default model
AVAILABLE_MODELS = {
    "gemini-2.5-pro-preview-05-06": {"temperature": 0.02},
    "gemini-2.5-flash-preview-04-17": {"temperature": 0.05}
}
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

LLM_TEMPERATURE = 0.02
NUM_RETRIEVED_DOCS = 30
RETRIEVAL_DISTANCE_THRESHOLD = 0.88

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# Global variables
embeddings_service = None
chat_model = None
chroma_client = None
collection = None
rag_workflow = None

# Define the comprehensive state for LangGraph workflow
class RAGState(TypedDict):
    query: str
    user_query_term: str
    query_type: Literal["numbered_definition", "field_lookup", "concept_search", "table_lookup", "general"]
    extracted_number: Optional[str]
    extracted_field_name: Optional[str]
    search_variations: List[str]
    retrieval_strategies: List[str]
    retrieved_docs: List[Dict]
    processed_docs: List[Dict]
    context_string: str
    final_answer: str
    confidence_score: float
    retry_count: int
    error_message: Optional[str]
    exact_matches: List[Dict]
    table_data: List[Dict]

# Field name to number mapping for common medical fields
FIELD_NAME_TO_NUMBER = {
    "hcfa admit type code": "141",
    "hcfa admit type": "141", 
    "admit type code": "141",
    "hcfa admit source code": "140",
    "hcfa admit source": "140",
    "admit source code": "140",
    "hcfa place of service": "139",
    "hcfa place of service code": "139",
    "place of service": "77",
    "pricing method code": "131",
    "pricing method": "131",
    "diagnosis code 6": "158",
    "diagnosis code 1": "153",
    "ahf_bfd_amt": "102",
    "aetna health fund before fund deductible": "102",
    "ahf before fund deductible": "102",
    "cob paid amount": "101",
    "coordination of benefits paid amount": "101",
    "paid amount": "100",
    "benefit payable": "99",
    "billed eligible amount": "123"
}

def init_clients_rag(): 
    global embeddings_service, chat_model, chroma_client, collection
    if not GCP_PROJECT_ID:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is essential and not set.")
    
    if not embeddings_service:
        embeddings_service = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME, project=GCP_PROJECT_ID)
    
    if not chroma_client:
        if not os.path.exists(CHROMA_DB_DIR):
            raise FileNotFoundError(f"ChromaDB directory '{CHROMA_DB_DIR}' missing. Run create_chroma_db.py first.")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    if not collection:
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            logger.info(f"Loaded Chroma collection '{COLLECTION_NAME}' ({collection.count()} items).")
        except Exception as e:
            logger.error(f"Error: Collection '{COLLECTION_NAME}' not found: {e}")
            raise

def get_chat_model(model_name=None):
    if not model_name or model_name not in AVAILABLE_MODELS:
        model_name = CHAT_MODEL_NAME  # Default model
    
    model_config = AVAILABLE_MODELS[model_name]
    temperature = model_config.get("temperature", LLM_TEMPERATURE)
    
    logger.info(f"Initializing ChatVertexAI with model: {model_name}, temperature: {temperature}")
    return ChatVertexAI(
        model_name=model_name, 
        project=GCP_PROJECT_ID, 
        temperature=temperature
    )

# Enhanced Query Analysis Node
def analyze_query(state: RAGState) -> RAGState:
    """Advanced query analysis with multi-pattern detection"""
    query = state["query"]
    
    # Initialize variables
    query_type = "general"
    user_query_term = query.strip()
    extracted_number = None
    extracted_field_name = None
    search_variations = []
    
    # Pattern 1: Direct numbered definition (e.g., "158. Diagnosis Code 6", "Diagnosis Code 6")
    numbered_def_patterns = [
        r"^(\d+)\.\s*([^:?\n]+)",  # "158. Diagnosis Code 6"
        r"(\d+)\.\s*([^:?\n]+):",  # "158. Diagnosis Code 6:"
        r"^(Field\s*)?#?\s*(\d+)[:\s]*(.*)",  # "Field #158: Diagnosis Code 6" or "#158 Diagnosis Code 6"
    ]
    
    for pattern in numbered_def_patterns:
        match = re.search(pattern, query.strip(), re.IGNORECASE)
        if match:
            query_type = "numbered_definition"
            if pattern == numbered_def_patterns[2] and match.group(2):  # Field pattern
                extracted_number = match.group(2)
                extracted_field_name = match.group(3).strip() if match.group(3) else ""
            else:
                extracted_number = match.group(1)
                extracted_field_name = match.group(2).strip()
            user_query_term = extracted_field_name or f"Field {extracted_number}"
            logger.info(f"Detected numbered definition: {extracted_number}. {extracted_field_name}")
            break
    
    # Pattern 2: Field name without number (e.g., "Diagnosis Code 6", "HCFA Admit Type Code")
    if query_type == "general":
        field_name_patterns = [
            r"^([A-Za-z][A-Za-z0-9\s\-_]+(?:Code|Amount|ID|Number|Date|Type|Name|Status|Category)?)\s*\d*$",
            r"(?:what\s+is\s+|define\s+|explain\s+)?([A-Za-z][A-Za-z0-9\s\-_]+(?:Code|Amount|ID|Number|Date|Type|Name|Status|Category))\s*\??$"
        ]
        
        for pattern in field_name_patterns:
            match = re.search(pattern, query.strip(), re.IGNORECASE)
            if match:
                query_type = "field_lookup"
                extracted_field_name = match.group(1).strip()
                
                # Check if we have a field mapping
                normalized_name = extracted_field_name.lower().strip()
                if normalized_name in FIELD_NAME_TO_NUMBER:
                    extracted_number = FIELD_NAME_TO_NUMBER[normalized_name]
                    query_type = "numbered_definition"
                    logger.info(f"Found field mapping: {extracted_field_name} -> {extracted_number}")
                
                user_query_term = extracted_field_name
                logger.info(f"Detected field lookup: {extracted_field_name}")
                break
    
    # Pattern 3: Technical names (e.g., "ICD9_DX_CD", "AHF_BFD_AMT")
    if query_type == "general":
        tech_name_pattern = r"^([A-Z][A-Z0-9_]{2,})$"
        match = re.search(tech_name_pattern, query.strip())
        if match:
            query_type = "field_lookup"
            extracted_field_name = match.group(1)
            user_query_term = extracted_field_name
            logger.info(f"Detected technical name: {extracted_field_name}")
    
    # Generate search variations based on detected patterns
    search_variations = generate_search_variations(user_query_term, extracted_number, extracted_field_name)
    
    # Clean up the term
    user_query_term = user_query_term.strip('\'"').strip()
    
    return {
        **state,
        "user_query_term": user_query_term,
        "query_type": query_type,
        "extracted_number": extracted_number,
        "extracted_field_name": extracted_field_name,
        "search_variations": search_variations,
        "retry_count": 0,
        "confidence_score": 0.0,
        "exact_matches": [],
        "table_data": []
    }

def generate_search_variations(term: str, number: Optional[str], field_name: Optional[str]) -> List[str]:
    """Generate comprehensive search variations for better retrieval"""
    variations = []
    
    if number and field_name:
        # For numbered definitions - prioritize exact matches
        variations.extend([
            f"{number}. {field_name}:",
            f"**{number}. {field_name}:**",
            f"{number}. {field_name}",
            f"Field #{number}",
            f"Field {number}",
            field_name,
            field_name.upper(),
            field_name.lower()
        ])
        
        # Add common medical field variations
        if "diagnosis code" in field_name.lower():
            variations.extend([
                field_name.replace("Diagnosis Code", "ICD"),
                field_name.replace("Code", ""),
                f"ICD Diagnosis {field_name.split()[-1] if field_name.split() else ''}"
            ])
        
        # Add HCFA variations
        if "hcfa" in field_name.lower():
            variations.extend([
                field_name.replace("HCFA", "").strip(),
                f"HCFA {field_name.split('HCFA')[-1].strip()}" if 'HCFA' in field_name else field_name
            ])
    
    elif field_name:
        # For field lookups without numbers
        variations.extend([
            field_name,
            field_name.upper(),
            field_name.lower(),
            field_name.title(),
            f"Technical Name: {field_name}",
            f"**{field_name}**"
        ])
        
        # Add acronym variations
        words = field_name.split()
        if len(words) > 1:
            acronym = ''.join(word[0].upper() for word in words if word)
            variations.append(acronym)
    
    else:
        variations = [term]
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(variations))

# Exact Match Detection Node
def detect_exact_matches(state: RAGState) -> RAGState:
    """Detect exact matches for numbered definitions"""
    search_variations = state["search_variations"]
    extracted_number = state["extracted_number"]
    
    exact_matches = []
    
    for search_term in search_variations[:5]:  # Limit to top 5 variations
        try:
            # Create embedding for search term
            query_embedding = embeddings_service.embed_query(search_term)
            
            # Search with high precision
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                    distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                    
                    # Check for exact numbered definition match
                    is_exact = False
                    if extracted_number:
                        # Multiple exact match patterns
                        exact_patterns = [
                            rf"^{extracted_number}\.\s*{re.escape(state['extracted_field_name'])}:",
                            rf"^\*\*{extracted_number}\.\s*{re.escape(state['extracted_field_name'])}:",
                            rf"^{extracted_number}\.\s*\[{re.escape(state['extracted_field_name'])}\]",
                        ]
                        
                        for pattern in exact_patterns:
                            if re.search(pattern, doc, re.IGNORECASE):
                                is_exact = True
                                logger.info(f"Found exact match: {extracted_number}. {state['extracted_field_name']}")
                                break
                    
                    if is_exact or distance < 0.1:  # Very close match
                        exact_matches.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'is_exact_match': is_exact,
                            'search_term': search_term
                        })
        
        except Exception as e:
            logger.error(f"Error in exact match detection: {e}")
    
    # Sort by exactness and distance
    exact_matches.sort(key=lambda x: (not x['is_exact_match'], x['distance']))
    
    return {
        **state,
        "exact_matches": exact_matches[:5]  # Keep top 5 exact matches
    }

# Enhanced Numbered Definition Retrieval Node
def numbered_definition_retrieval(state: RAGState) -> RAGState:
    """Enhanced retrieval specifically for numbered definitions"""
    search_variations = state["search_variations"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    all_results = []
    seen_docs = set()
    
    # Use search variations for comprehensive retrieval
    for search_query in search_variations:
        try:
            query_embedding = embeddings_service.embed_query(search_query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=8,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    if doc not in seen_docs:
                        seen_docs.add(doc)
                        metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                        distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                        
                        # Enhanced matching logic
                        is_exact_match = False
                        found_number = None
                        contains_definition = False
                        
                        if extracted_number:
                            # Check for exact numbered definition
                            exact_patterns = [
                                rf"^{extracted_number}\.\s*{re.escape(extracted_field_name)}:",
                                rf"^\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}:",
                                rf"^{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]"
                            ]
                            
                            for pattern in exact_patterns:
                                if re.search(pattern, doc, re.IGNORECASE):
                                    is_exact_match = True
                                    found_number = extracted_number
                                    break
                            
                            # Check for partial match with correct number
                            if not is_exact_match:
                                number_match = re.search(rf"(\d+)\.\s*.*{re.escape(extracted_field_name.split()[0])}", doc, re.IGNORECASE)
                                if number_match:
                                    found_number = number_match.group(1)
                        
                        # Check if contains definition elements
                        definition_indicators = ['Format:', 'Technical Name:', 'Length:', 'Position', 'Definition:']
                        contains_definition = sum(1 for indicator in definition_indicators if indicator in doc) >= 3
                        
                        all_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'is_exact_match': is_exact_match,
                            'found_number': found_number,
                            'contains_definition': contains_definition,
                            'retrieval_strategy': 'numbered_definition',
                            'search_query': search_query
                        })
        
        except Exception as e:
            logger.error(f"Error in numbered definition search: {e}")
    
    # Enhanced sorting: exact matches first, then complete definitions, then by distance
    all_results.sort(key=lambda x: (
        not x.get('is_exact_match', False),
        not x.get('contains_definition', False),
        x['distance'],
        -len(x['content'])
    ))
    
    return {
        **state,
        "retrieved_docs": state.get("retrieved_docs", []) + all_results[:15],
        "retrieval_strategies": state.get("retrieval_strategies", []) + ["numbered_definition"]
    }

# Enhanced Field Lookup Retrieval Node
def field_lookup_retrieval(state: RAGState) -> RAGState:
    """Enhanced field lookup with better technical term matching"""
    search_variations = state["search_variations"]
    extracted_field_name = state["extracted_field_name"]
    
    all_results = []
    seen_docs = set()
    
    # Enhanced search strategies for field lookup
    enhanced_queries = search_variations.copy()
    
    # Add technical name variations
    if extracted_field_name:
        enhanced_queries.extend([
            f"Technical Name: {extracted_field_name}",
            f"Technical Name: {extracted_field_name.upper()}",
            extracted_field_name.replace(' ', '_').upper(),  # Convert to tech name format
            extracted_field_name.replace(' ', ''),  # Remove spaces
        ])
    
    for search_query in enhanced_queries:
        try:
            query_embedding = embeddings_service.embed_query(search_query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=6,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    if doc not in seen_docs:
                        seen_docs.add(doc)
                        metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                        distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                        
                        # Enhanced field matching
                        is_field_match = False
                        is_tech_name_match = False
                        is_numbered_definition = False
                        
                        # Check for technical name match
                        tech_name_patterns = [
                            rf"Technical Name:\s*{re.escape(extracted_field_name)}",
                            rf"Technical Name:\s*{re.escape(extracted_field_name.upper())}",
                            rf"Technical Name:\s*{re.escape(extracted_field_name.replace(' ', '_').upper())}"
                        ]
                        
                        for pattern in tech_name_patterns:
                            if re.search(pattern, doc, re.IGNORECASE):
                                is_tech_name_match = True
                                break
                        
                        # Check for field title match
                        field_patterns = [
                            rf"^\d+\.\s*{re.escape(extracted_field_name)}:",
                            rf"^\*\*.*{re.escape(extracted_field_name)}.*\*\*",
                            rf"^{re.escape(extracted_field_name)}:"
                        ]
                        
                        for pattern in field_patterns:
                            if re.search(pattern, doc, re.IGNORECASE):
                                is_field_match = True
                                # Check if it's a numbered definition
                                if re.search(rf"^\d+\.", doc):
                                    is_numbered_definition = True
                                break
                        
                        all_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'is_field_match': is_field_match,
                            'is_tech_name_match': is_tech_name_match,
                            'is_numbered_definition': is_numbered_definition,
                            'retrieval_strategy': 'field_lookup',
                            'search_query': search_query
                        })
        
        except Exception as e:
            logger.error(f"Error in field lookup search: {e}")
    
    # Sort by match quality
    all_results.sort(key=lambda x: (
        not (x.get('is_tech_name_match', False) or x.get('is_field_match', False)),
        not x.get('is_numbered_definition', False),
        x['distance']
    ))
    
    return {
        **state,
        "retrieved_docs": state.get("retrieved_docs", []) + all_results[:12],
        "retrieval_strategies": state.get("retrieval_strategies", []) + ["field_lookup"]
    }

# Table and Appendix Retrieval Node
def table_appendix_retrieval(state: RAGState) -> RAGState:
    """Retrieve related tables and appendix information"""
    user_query_term = state["user_query_term"]
    extracted_number = state["extracted_number"]
    
    # Search for related tables and appendices
    table_queries = [
        f"{user_query_term} values",
        f"{user_query_term} codes",
        f"{user_query_term} appendix",
        f"Field {extracted_number} values" if extracted_number else "",
        f"appendix {user_query_term}",
        f"values and definitions {user_query_term}"
    ]
    
    table_results = []
    
    for query in table_queries:
        if not query:
            continue
        
        try:
            query_embedding = embeddings_service.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"],
                where={"$or": [
                    {"is_embedded_table": True},
                    {"content_type": "appendix"},
                    {"content_type": "value_mapping"}
                ]}
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                    distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                    
                    if distance < 0.9:  # Only include relevant tables
                        table_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'retrieval_strategy': 'table_appendix',
                            'search_query': query
                        })
        
        except Exception as e:
            logger.error(f"Error in table retrieval: {e}")
    
    # Sort by relevance
    table_results.sort(key=lambda x: x['distance'])
    
    return {
        **state,
        "table_data": table_results[:5],
        "retrieval_strategies": state.get("retrieval_strategies", []) + ["table_appendix"]
    }

# Enhanced Semantic Retrieval Node
def semantic_retrieval(state: RAGState) -> RAGState:
    """Semantic retrieval with query expansion"""
    user_query_term = state["user_query_term"]
    query = state["query"]
    query_type = state["query_type"]
    
    # Enhanced query rewriting for medical terminology
    rewrite_prompt = f"""
    Optimize the following query for semantic search in a medical data dictionary.
    The dictionary contains numbered field definitions, technical specifications, format details, and appendices.
    
    Original query: "{query}"
    Core term: "{user_query_term}"
    Query type: {query_type}
    
    Create 3 search variations that would find relevant medical field definitions and specifications.
    Focus on medical terminology, field names, codes, and data definitions.
    
    Return only the 3 optimized queries, one per line, without numbering or additional text.
    """
    
    optimized_queries = [user_query_term]  # Default fallback
    
    try:
        response = chat_model.invoke(rewrite_prompt).content.strip()
        optimized_queries = [q.strip() for q in response.split('\n') if q.strip()]
        if not optimized_queries:
            optimized_queries = [user_query_term]
    except Exception as e:
        logger.error(f"Error in query rewriting: {e}")
        optimized_queries = [user_query_term]
    
    # Add original search variations
    optimized_queries.extend(state.get("search_variations", [])[:3])
    
    all_results = []
    
    for query_text in optimized_queries[:5]:  # Limit to 5 queries
        try:
            query_embedding = embeddings_service.embed_query(query_text)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                    distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                    
                    # Skip if already retrieved
                    existing_docs = [d['content'] for d in state.get("retrieved_docs", [])]
                    if doc not in existing_docs:
                        all_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'retrieval_strategy': 'semantic',
                            'search_query': query_text
                        })
        
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
    
    # Sort by distance
    all_results.sort(key=lambda x: x['distance'])
    
    return {
        **state,
        "retrieved_docs": state.get("retrieved_docs", []) + all_results[:10],
        "retrieval_strategies": state.get("retrieval_strategies", []) + ["semantic"]
    }

# Enhanced Document Processing and Ranking Node
def process_and_rank_documents(state: RAGState) -> RAGState:
    """Enhanced document processing with sophisticated scoring"""
    retrieved_docs = state["retrieved_docs"]
    exact_matches = state["exact_matches"]
    table_data = state["table_data"]
    user_query_term = state["user_query_term"]
    query = state["query"]
    query_type = state["query_type"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    # Combine all retrieved documents
    all_docs = []
    
    # Add exact matches with highest priority
    for match in exact_matches:
        all_docs.append({**match, 'source': 'exact_match'})
    
    # Add regular retrieved docs
    for doc in retrieved_docs:
        # Avoid duplicates
        if doc['content'] not in [d['content'] for d in all_docs]:
            all_docs.append({**doc, 'source': 'retrieval'})
    
    # Add table data
    for table in table_data:
        all_docs.append({**table, 'source': 'table_data'})
    
    processed_docs = []
    
    for doc in all_docs:
        content = doc['content']
        metadata = doc['metadata']
        distance = doc['distance']
        
        # Initialize comprehensive relevance score
        relevance_score = 0.0
        
        # 1. Source-based scoring
        if doc.get('source') == 'exact_match':
            relevance_score += 150.0
        elif doc.get('source') == 'table_data':
            relevance_score += 50.0
        
        # 2. Exact match pattern scoring
        if query_type == "numbered_definition" and extracted_number and extracted_field_name:
            # Multiple exact match patterns with different weights
            exact_patterns = [
                (rf"^{extracted_number}\.\s*{re.escape(extracted_field_name)}:", 200.0),
                (rf"^\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}:", 180.0),
                (rf"^{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]", 160.0),
                (rf"^{extracted_number}\.\s*{re.escape(extracted_field_name)}", 140.0),
                # More flexible patterns
                (rf"{extracted_number}\.\s*{re.escape(extracted_field_name)}", 120.0),
                (rf"\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}", 100.0),
            ]
            
            for pattern, score in exact_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    relevance_score += score
                    logger.info(f"  *** EXACT PATTERN MATCH *** {extracted_number}. {extracted_field_name}")
                    break
        
        # 3. Technical name exact match (very high value)
        if extracted_field_name:
            tech_name_pattern = rf"Technical Name:\s*{re.escape(extracted_field_name.replace(' ', '_').upper())}"
            if re.search(tech_name_pattern, content, re.IGNORECASE):
                relevance_score += 120.0
        
        # 4. Complete definition structure scoring
        definition_elements = ['Format:', 'Technical Name:', 'Length:', 'Position', 'Definition:']
        element_count = sum(1 for element in definition_elements if element in content)
        relevance_score += element_count * 15.0
        
        # Bonus for complete definitions
        if element_count >= 4:
            relevance_score += 30.0
        
        # 5. Metadata-based scoring (with string handling)
        def safe_bool_convert(value, default=False):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', 'yes', '1']
            return default
        
        def safe_int_convert(value, default=0):
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return default
        
        # Metadata indicators
        if safe_bool_convert(metadata.get("is_numbered_definition")):
            relevance_score += 40.0
        if safe_bool_convert(metadata.get("is_complete_definition")):
            relevance_score += 35.0
        if safe_bool_convert(metadata.get("is_field_spec")):
            relevance_score += 25.0
        
        # Definition completeness score
        completeness = safe_int_convert(metadata.get("definition_completeness", 0))
        relevance_score += completeness * 8.0
        
        # 6. Field number matching
        if extracted_number:
            meta_number = metadata.get("definition_number", "")
            if meta_number == extracted_number:
                relevance_score += 60.0
        
        # 7. Content quality indicators
        content_length = len(content)
        if content_length > 200:  # Substantial content
            relevance_score += 10.0
        if content_length > 500:  # Very detailed content
            relevance_score += 20.0
        
        # 8. Distance-based scoring (inverse relationship)
        distance_score = max(0, 30.0 * (1.0 - distance))
        relevance_score += distance_score
        
        # 9. Search strategy bonus
        if doc.get('is_exact_match'):
            relevance_score += 80.0
        if doc.get('is_tech_name_match'):
            relevance_score += 60.0
        if doc.get('is_field_match'):
            relevance_score += 50.0
        
        # 10. Content type specific scoring
        content_type = metadata.get('content_type', '')
        if content_type == 'numbered_definition':
            relevance_score += 25.0
        elif content_type == 'field_specification':
            relevance_score += 20.0
        elif content_type == 'value_mapping':
            relevance_score += 15.0
        
        # 11. Table and appendix bonuses
        if safe_bool_convert(metadata.get("has_embedded_tables")):
            relevance_score += 15.0
        if content_type == 'appendix':
            relevance_score += 10.0
        
        # 12. Penalty for incomplete or very short content
        if content_length < 50:
            relevance_score -= 15.0
        
        # Calculate confidence score
        confidence = min(1.0, relevance_score / 200.0)
        
        processed_docs.append({
            **doc,
            'relevance_score': relevance_score,
            'confidence': confidence,
            'quality_indicators': {
                'has_definition_structure': element_count >= 3,
                'is_complete': safe_bool_convert(metadata.get("is_complete_definition")),
                'content_length': content_length,
                'definition_elements': element_count,
                'contains_technical_name': 'Technical Name:' in content,
                'contains_format': 'Format:' in content,
                'contains_definition': 'Definition:' in content
            }
        })
    
    # Sort by relevance score (descending)
    processed_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Calculate overall confidence
    top_score = processed_docs[0]['relevance_score'] if processed_docs else 0
    confidence_score = min(1.0, top_score / 200.0)
    
    # Log ranking information for debugging
    logger.info("\n--- Document Ranking Results ---")
    for i, doc in enumerate(processed_docs[:5]):
        metadata = doc['metadata']
        logger.info(f"{i+1}. Score: {doc['relevance_score']:.1f}, Distance: {doc['distance']:.3f}, Confidence: {doc['confidence']:.2f}")
        logger.info(f"   Strategy: {doc.get('retrieval_strategy', 'unknown')}, Source: {doc.get('source', 'unknown')}")
        logger.info(f"   Content preview: {doc['content'][:100]}...")
        
        # Check if content contains the field number we're looking for
        content = doc['content']
        if extracted_number:
            if f"{extracted_number}." in content:
                logger.info(f"   *** Contains field {extracted_number} ***")
                # Extract a snippet around the field number
                pattern = rf"({extracted_number}\..*?)(?=\n\d+\.|$)"
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    snippet = match.group(1)[:200] + "..." if len(match.group(1)) > 200 else match.group(1)
                    logger.info(f"   Snippet: {snippet}")
        
        if doc.get('found_number'):
            logger.info(f"   Found field number: {doc['found_number']}")
    
    return {
        **state,
        "processed_docs": processed_docs,
        "confidence_score": confidence_score
    }

# Enhanced Context Building Node
def build_context(state: RAGState) -> RAGState:
    """Build comprehensive context with proper formatting"""
    processed_docs = state["processed_docs"]
    query_type = state["query_type"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    if not processed_docs:
        return {
            **state,
            "context_string": "No relevant information found.",
            "confidence_score": 0.0
        }
    
    # Filter documents by quality and relevance
    high_quality_docs = [
        doc for doc in processed_docs 
        if doc['distance'] <= RETRIEVAL_DISTANCE_THRESHOLD and doc['relevance_score'] > 20.0
    ]
    
    if not high_quality_docs:
        # Relax criteria if no high-quality docs found
        high_quality_docs = [
            doc for doc in processed_docs 
            if doc['distance'] <= 0.95 and doc['relevance_score'] > 10.0
        ][:3]
    
    if not high_quality_docs:
        return {
            **state,
            "context_string": "No relevant information found matching quality criteria.",
            "confidence_score": 0.0
        }
    
    # For numbered definitions, prioritize exact matches and complete definitions
    if query_type == "numbered_definition" and extracted_number:
        # Separate exact matches from others
        exact_matches = [doc for doc in high_quality_docs if doc.get('relevance_score', 0) > 150]
        other_docs = [doc for doc in high_quality_docs if doc.get('relevance_score', 0) <= 150]
        
        # Prioritize exact matches
        final_docs = exact_matches[:2] + other_docs[:3]
    else:
        final_docs = high_quality_docs[:5]
    
    # Build context with detailed source attribution
    context_parts = []
    
    for i, doc in enumerate(final_docs):
        metadata = doc['metadata']
        content = doc['content']
        source = metadata.get('source_filename', 'Unknown source')
        page = metadata.get('page_number', 'N/A')
        element_type = metadata.get('element_type', 'text')
        relevance_score = doc.get('relevance_score', 0)
        retrieval_strategy = doc.get('retrieval_strategy', 'unknown')
        
        # Clean up the content for better presentation
        content = re.sub(r'\n+', '\n', content)
        content = content.strip()
        
        # Add context header with metadata
        context_header = f"[Source: {source}, Page: {page}, Type: {element_type}, Score: {relevance_score:.1f}, Strategy: {retrieval_strategy}]"
        context_parts.append(f"{context_header}\n{content}")
    
    context_string = "\n\n---\n\n".join(context_parts)
    
    return {
        **state,
        "context_string": context_string
    }

# Enhanced Answer Generation Node
def generate_answer(state: RAGState) -> RAGState:
    """Generate properly formatted answer matching expected output format"""
    context_string = state["context_string"]
    query = state["query"]
    user_query_term = state["user_query_term"]
    query_type = state["query_type"]
    confidence_score = state["confidence_score"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    if not context_string or context_string == "No relevant information found.":
        return {
            **state,
            "final_answer": "I could not find relevant information for your query in the document.",
            "confidence_score": 0.0
        }
    
    # Create specialized prompt for numbered definitions
    if query_type == "numbered_definition" and extracted_number:
        prompt_template = """You are an AI assistant that extracts numbered field definitions from a medical data dictionary.

The user asked about: "{original_question}"
Looking for field number: {field_number}
Field name: "{field_name}"

CRITICAL INSTRUCTIONS:
1. Find the COMPLETE numbered definition that starts with "{field_number}. {field_name}" in the context
2. Extract ALL the field details and format them EXACTLY as shown below
3. The output must follow this EXACT format:

**{field_number}. {field_name}:**
• **Format:** [extract the actual format value from the context]
• **Technical Name:** [extract the actual technical name from the context]
• **Length:** [extract the actual length value from the context]
• **Positions:** [extract the actual positions value from the context]
• **Definition:** [extract the complete definition text from the context]

4. IMPORTANT: You must extract the ACTUAL VALUES from the context, not placeholders
5. Look for patterns like "Format: Character", "Technical Name: HCFA_ADMIT_TYPE_CD", etc.
6. Include any additional notes, especially those starting with "NOTE:"
7. If you cannot find the exact field, look for any content related to the field name

Context from the medical data dictionary:
---
{context}
---

Extract and format the numbered definition with the ACTUAL VALUES from the context:"""
        
        final_prompt = prompt_template.format(
            original_question=query,
            field_number=extracted_number,
            field_name=extracted_field_name,
            context=context_string
        )
        
    elif query_type == "field_lookup":
        prompt_template = """You are an AI assistant that explains fields from a medical data dictionary.

The user asked about: "{original_question}"
Looking for field: "{field_name}"

INSTRUCTIONS:
1. Find the complete definition for this field
2. If it's a numbered definition, use this format:

**[NUMBER]. [FIELD NAME]:**
• **Format:** [value]
• **Technical Name:** [value]
• **Length:** [value]
• **Positions:** [value]
• **Definition:** [complete definition]

3. If it's not numbered, present all available information clearly
4. Include any embedded tables, value mappings, or related information
5. Preserve exact formatting and spacing
6. Include any notes or special information

Context from the medical data dictionary:
---
{context}
---

Provide the complete field information with proper formatting:"""
        
        final_prompt = prompt_template.format(
            original_question=query,
            field_name=extracted_field_name or user_query_term,
            context=context_string
        )
    
    else:  # General queries
        prompt_template = """You are an AI assistant that explains concepts from a medical data dictionary.

The user asked about: "{original_question}"
Topic: "{topic}"

INSTRUCTIONS:
1. Provide a comprehensive explanation based on the context
2. If the context contains numbered definitions, present them with proper formatting:

**[NUMBER]. [FIELD NAME]:**
• **Format:** [value]
• **Technical Name:** [value]
• **Length:** [value]
• **Positions:** [value]
• **Definition:** [definition]

3. Include any relevant tables, value mappings, or cross-references
4. Organize information logically and clearly
5. Preserve exact formatting from the source

Context from the medical data dictionary:
---
{context}
---

Provide a comprehensive explanation with proper formatting:"""
        
        final_prompt = prompt_template.format(
            original_question=query,
            topic=user_query_term,
            context=context_string
        )
    
    # Generate the answer
    try:
        response = chat_model.invoke(final_prompt)
        final_answer = response.content.strip()
        
        # Post-process the answer for better formatting
        final_answer = post_process_answer(final_answer, query_type, extracted_number)
        
        # Adjust confidence based on answer quality
        if "could not find" in final_answer.lower() or "no information" in final_answer.lower():
            confidence_score *= 0.3
        elif extracted_number and f"{extracted_number}." in final_answer:
            confidence_score = min(1.0, confidence_score + 0.2)
        
        return {
            **state,
            "final_answer": final_answer,
            "confidence_score": confidence_score
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            **state,
            "final_answer": f"Error generating response: {e}",
            "confidence_score": 0.0
        }

def post_process_answer(answer: str, query_type: str, extracted_number: Optional[str]) -> str:
    """Post-process the answer to ensure proper formatting"""
    # Ensure proper bullet point formatting
    answer = re.sub(r'^[\s-]*\*\s*\*\*([^:]+):\*\*', r'• **\1:**', answer, flags=re.MULTILINE)
    answer = re.sub(r'^[\s-]*\*\s*([^:]*:)', r'• **\1**', answer, flags=re.MULTILINE)
    
    # Ensure proper numbered definition header
    if query_type == "numbered_definition" and extracted_number:
        # Make sure the header is properly formatted
        header_pattern = rf'^{extracted_number}\.([^:]+):'
        header_replacement = rf'**{extracted_number}.\1:**'
        answer = re.sub(header_pattern, header_replacement, answer, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
    
    return answer.strip()

# Validation Node
def validate_answer(state: RAGState) -> RAGState:
    """Validate the quality of the generated answer"""
    final_answer = state["final_answer"]
    query_type = state["query_type"]
    user_query_term = state["user_query_term"]
    confidence_score = state["confidence_score"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    validation_score = confidence_score
    
    # Enhanced validation for numbered definitions
    if query_type == "numbered_definition":
        # Check for proper numbered definition format
        if extracted_number and f"**{extracted_number}." in final_answer:
            validation_score += 0.3
        if extracted_field_name and extracted_field_name in final_answer:
            validation_score += 0.2
        
        # Check for complete definition structure
        definition_elements = ['Format:', 'Technical Name:', 'Length:', 'Position', 'Definition:']
        element_count = sum(1 for element in definition_elements if element in final_answer)
        validation_score += (element_count / len(definition_elements)) * 0.4
        
        # Check for proper bullet point formatting
        bullet_points = len(re.findall(r'• \*\*[^:]+:\*\*', final_answer))
        if bullet_points >= 4:
            validation_score += 0.2
    
    elif query_type == "field_lookup":
        # Check for field-specific information
        if extracted_field_name and extracted_field_name in final_answer:
            validation_score += 0.2
        if 'Technical Name:' in final_answer or 'Format:' in final_answer:
            validation_score += 0.3
        if re.search(r'\*\*\d+\.', final_answer):  # Found numbered definition
            validation_score += 0.2
    
    # General quality checks
    if len(final_answer) > 150:  # Substantial answer
        validation_score += 0.1
    if final_answer.count('•') >= 3:  # Well-structured with bullet points
        validation_score += 0.1
    if 'NOTE:' in final_answer:  # Includes important notes
        validation_score += 0.05
    
    # Cap the validation score
    validation_score = min(1.0, validation_score)
    
    return {
        **state,
        "confidence_score": validation_score
    }

# Route selection functions
def route_retrieval_strategy(state: RAGState) -> str:
    """Determine which retrieval strategy to use"""
    query_type = state["query_type"]
    
    if query_type == "numbered_definition":
        return "numbered_definition"
    elif query_type == "field_lookup":
        return "field_lookup"
    elif query_type == "table_lookup":
        return "table_appendix"
    else:
        return "semantic"

def route_validation(state: RAGState) -> str:
    """Determine if answer needs retry"""
    confidence_score = state["confidence_score"]
    retry_count = state["retry_count"]
    
    if confidence_score < 0.4 and retry_count < 2:
        return "retry"
    else:
        return "end"

# Retry logic node
def retry_with_fallback(state: RAGState) -> RAGState:
    """Retry with different strategy"""
    retry_count = state["retry_count"]
    query_type = state["query_type"]
    
    new_retry_count = retry_count + 1
    
    # Fallback strategy
    fallback_mapping = {
        "numbered_definition": "field_lookup" if retry_count == 0 else "semantic",
        "field_lookup": "numbered_definition" if retry_count == 0 else "semantic",
        "table_lookup": "semantic",
        "semantic": "field_lookup"
    }
    
    new_query_type = fallback_mapping.get(query_type, "semantic")
    
    logger.info(f"Retrying with strategy: {new_query_type} (attempt {new_retry_count})")
    
    return {
        **state,
        "query_type": new_query_type,
        "retry_count": new_retry_count,
        "retrieved_docs": [],
        "processed_docs": [],
        "exact_matches": [],
        "table_data": [],
        "context_string": "",
        "confidence_score": 0.0
    }

# Fallback pipeline
def fallback_rag_pipeline(state: RAGState) -> RAGState:
    """Fallback implementation when LangGraph is not available"""
    logger.info("Using fallback RAG pipeline...")
    
    # Execute pipeline steps
    state = analyze_query(state)
    state = detect_exact_matches(state)
    
    query_type = state["query_type"]
    if query_type == "numbered_definition":
        state = numbered_definition_retrieval(state)
    elif query_type == "field_lookup":
        state = field_lookup_retrieval(state)
    else:
        state = semantic_retrieval(state)
    
    # Always try to get table data for completeness
    state = table_appendix_retrieval(state)
    state = process_and_rank_documents(state)
    state = build_context(state)
    state = generate_answer(state)
    state = validate_answer(state)
    
    # Simple retry logic
    if state["confidence_score"] < 0.4 and state["retry_count"] < 1:
        logger.info("Low confidence, attempting retry...")
        state = retry_with_fallback(state)
        state = semantic_retrieval(state)
        state = process_and_rank_documents(state)
        state = build_context(state)
        state = generate_answer(state)
        state = validate_answer(state)
    
    return state

def create_rag_workflow() -> CompiledGraph:
    """Create the comprehensive RAG workflow with LangGraph"""
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, using fallback")
        return FallbackGraph()
    
    workflow = StateGraph(RAGState)
    
    # Add all nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("detect_exact_matches", detect_exact_matches)
    workflow.add_node("numbered_definition_retrieval", numbered_definition_retrieval)
    workflow.add_node("field_lookup_retrieval", field_lookup_retrieval)
    workflow.add_node("table_appendix_retrieval", table_appendix_retrieval)
    workflow.add_node("semantic_retrieval", semantic_retrieval)
    workflow.add_node("process_and_rank", process_and_rank_documents)
    workflow.add_node("build_context", build_context)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("validate_answer", validate_answer)
    workflow.add_node("retry_with_fallback", retry_with_fallback)
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    # Flow from analysis to exact match detection
    workflow.add_edge("analyze_query", "detect_exact_matches")
    
    # Route from exact match detection to appropriate retrieval
    workflow.add_conditional_edges(
        "detect_exact_matches",
        route_retrieval_strategy,
        {
            "numbered_definition": "numbered_definition_retrieval",
            "field_lookup": "field_lookup_retrieval",
            "table_appendix": "table_appendix_retrieval",
            "semantic": "semantic_retrieval"
        }
    )
    
    # All retrieval strategies flow to table appendix retrieval
    workflow.add_edge("numbered_definition_retrieval", "table_appendix_retrieval")
    workflow.add_edge("field_lookup_retrieval", "table_appendix_retrieval")
    workflow.add_edge("semantic_retrieval", "table_appendix_retrieval")
    
    # Continue the workflow
    workflow.add_edge("table_appendix_retrieval", "process_and_rank")
    workflow.add_edge("process_and_rank", "build_context")
    workflow.add_edge("build_context", "generate_answer")
    workflow.add_edge("generate_answer", "validate_answer")
    
    # Validation routing
    workflow.add_conditional_edges(
        "validate_answer",
        route_validation,
        {
            "retry": "retry_with_fallback",
            "end": END
        }
    )
    
    # Retry routing
    workflow.add_conditional_edges(
        "retry_with_fallback",
        route_retrieval_strategy,
        {
            "numbered_definition": "numbered_definition_retrieval",
            "field_lookup": "field_lookup_retrieval",
            "table_appendix": "table_appendix_retrieval",
            "semantic": "semantic_retrieval"
        }
    )
    
    return workflow.compile()

# Initialize system
try:
    init_clients_rag()
    rag_workflow = create_rag_workflow()
    logger.info("Enhanced RAG workflow with LangGraph initialized successfully")
except Exception as e:
    logger.error(f"FATAL: Error during RAG initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/')
def index_route(): 
    return render_template('chat.html')

@app.route('/chat_rag', methods=['POST']) 
def chat_handler_rag():
    """Enhanced chat handler with comprehensive numbered definition support"""
    if not rag_workflow:
        return jsonify({"error": "RAG workflow not initialized."}), 500

    data = request.get_json()
    user_query = data.get('query')
    model_name = data.get('model')
    
    if not user_query: 
        return jsonify({"error": "Query not provided"}), 400

    try:
        # Get appropriate chat model
        current_chat_model = get_chat_model(model_name)
        
        # Initialize comprehensive state
        initial_state = RAGState(
            query=user_query,
            user_query_term="",
            query_type="general",
            extracted_number=None,
            extracted_field_name=None,
            search_variations=[],
            retrieval_strategies=[],
            retrieved_docs=[],
            processed_docs=[],
            context_string="",
            final_answer="",
            confidence_score=0.0,
            retry_count=0,
            error_message=None,
            exact_matches=[],
            table_data=[]
        )
        
        # Set the chat model globally for this request
        global chat_model
        chat_model = current_chat_model
        
        # Run the comprehensive workflow
        logger.info(f"Processing query with {model_name or CHAT_MODEL_NAME}: '{user_query}'")
        final_state = rag_workflow.invoke(initial_state)
        
        # Extract comprehensive results
        final_answer = final_state.get("final_answer", "No answer generated")
        confidence_score = final_state.get("confidence_score", 0.0)
        retrieval_strategies = final_state.get("retrieval_strategies", [])
        query_type = final_state.get("query_type", "unknown")
        extracted_number = final_state.get("extracted_number")
        extracted_field_name = final_state.get("extracted_field_name")
        
        # Get source metadata from processed docs
        source_metadata = []
        for doc in final_state.get("processed_docs", [])[:5]:
            metadata = doc.get("metadata", {})
            source_metadata.append({
                "source": metadata.get("source_filename", "Unknown"),
                "page": metadata.get("page_number", "N/A"),
                "type": metadata.get("element_type", "text"),
                "score": doc.get("relevance_score", 0.0),
                "distance": doc.get("distance", 1.0),
                "strategy": doc.get("retrieval_strategy", "unknown")
            })
        
        # Additional debug information
        debug_info = {
            "search_variations": final_state.get("search_variations", []),
            "exact_matches_count": len(final_state.get("exact_matches", [])),
            "table_data_count": len(final_state.get("table_data", [])),
            "total_retrieved": len(final_state.get("retrieved_docs", [])),
            "processed_docs": len(final_state.get("processed_docs", []))
        }
        
        logger.info(f"Query processed successfully. Type: {query_type}, Number: {extracted_number}, Field: {extracted_field_name}")
        logger.info(f"Confidence: {confidence_score:.2f}, Strategies: {retrieval_strategies}")
        
        return jsonify({
            "answer": final_answer,
            "confidence_score": confidence_score,
            "retrieval_strategies": retrieval_strategies,
            "retrieved_sources_metadata": source_metadata,
            "query_type": query_type,
            "extracted_number": extracted_number,
            "extracted_field_name": extracted_field_name,
            "model_used": model_name or CHAT_MODEL_NAME,
            "debug_info": debug_info
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            "error": "An internal error occurred. Please check server logs.",
            "query": user_query,
            "model": model_name
        }), 500

@app.route('/chat', methods=['POST'])
def chat_handler_fallback():
    return chat_handler_rag()

if __name__ == '__main__':
    logger.info("Starting Enhanced RAG Flask application with Advanced LangGraph...")
    logger.info(f"GOOGLE_CLOUD_PROJECT: {GCP_PROJECT_ID}")
    
    if not GCP_PROJECT_ID:
        logger.warning("GOOGLE_CLOUD_PROJECT not set. The application may not work properly.")
    
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)