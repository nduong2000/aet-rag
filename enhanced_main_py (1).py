# Enhanced RAG system with Advanced LangGraph workflow
# Fixed for proper numbered definition extraction and formatting

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

# Enhanced field name to number mapping for common medical fields
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

# Enhanced Query Analysis Node with better numbered definition detection
def analyze_query(state: RAGState) -> RAGState:
    """Advanced query analysis with improved numbered definition detection"""
    query = state["query"]
    
    # Initialize variables
    query_type = "general"
    user_query_term = query.strip()
    extracted_number = None
    extracted_field_name = None
    search_variations = []
    
    # Enhanced patterns for numbered definitions - more comprehensive
    numbered_def_patterns = [
        r"^(\d+)\.\s*([^:?\n]+)",          # "141. HCFA Admit Type Code"
        r"(\d+)\.\s*([^:?\n]+):",         # "141. HCFA Admit Type Code:"
        r"^(Field\s*)?#?\s*(\d+)[:\s]*(.*)",  # "Field #141: HCFA Admit Type Code"
        r"^\*\*(\d+)\.\s*([^*:?\n]+)\*\*",    # "**141. HCFA Admit Type Code**"
        r"(\d+)\.\s*\[([^\]]+)\]",        # "141. [HCFA Admit Type Code]"
    ]
    
    for i, pattern in enumerate(numbered_def_patterns):
        match = re.search(pattern, query.strip(), re.IGNORECASE)
        if match:
            query_type = "numbered_definition"
            if i == 2 and match.group(2):  # Field pattern
                extracted_number = match.group(2)
                extracted_field_name = match.group(3).strip() if match.group(3) else ""
            else:
                extracted_number = match.group(1)
                extracted_field_name = match.group(2).strip()
            
            # Clean up field name from formatting
            if extracted_field_name:
                extracted_field_name = re.sub(r'[\*\[\]_]', '', extracted_field_name).strip()
            
            user_query_term = extracted_field_name or f"Field {extracted_number}"
            logger.info(f"Detected numbered definition: {extracted_number}. {extracted_field_name}")
            break
    
    # Enhanced field name patterns with better coverage
    if query_type == "general":
        field_name_patterns = [
            r"^([A-Za-z][A-Za-z0-9\s\-_]+(?:Code|Amount|ID|Number|Date|Type|Name|Status|Category)?)\s*\d*$",
            r"(?:what\s+is\s+|define\s+|explain\s+)?([A-Za-z][A-Za-z0-9\s\-_]+(?:Code|Amount|ID|Number|Date|Type|Name|Status|Category))\s*\??$",
            r"([A-Za-z][A-Za-z0-9\s\-_]*(?:HCFA|Code|Type|Amount))\s*$",  # Better HCFA pattern
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
    
    # Technical names pattern
    if query_type == "general":
        tech_name_pattern = r"^([A-Z][A-Z0-9_]{2,})$"
        match = re.search(tech_name_pattern, query.strip())
        if match:
            query_type = "field_lookup"
            extracted_field_name = match.group(1)
            user_query_term = extracted_field_name
            logger.info(f"Detected technical name: {extracted_field_name}")
    
    # Generate enhanced search variations
    search_variations = generate_enhanced_search_variations(user_query_term, extracted_number, extracted_field_name)
    
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

def generate_enhanced_search_variations(term: str, number: Optional[str], field_name: Optional[str]) -> List[str]:
    """Generate comprehensive search variations with enhanced patterns"""
    variations = []
    
    if number and field_name:
        # Enhanced patterns for numbered definitions
        variations.extend([
            f"{number}. {field_name}:",
            f"**{number}. {field_name}:**",
            f"{number}. {field_name}",
            f"**{number}. [{field_name}]{{{.underline}}}:**",  # Underlined pattern
            f"[{field_name}]{{{.underline}}}:",  # Just underlined title
            f"Field #{number}",
            f"Field {number}",
            field_name,
            field_name.upper(),
            field_name.lower(),
            # Add variations without punctuation
            f"{number} {field_name}",
            f"**{number} {field_name}**",
        ])
        
        # Special handling for HCFA codes
        if "hcfa" in field_name.lower():
            variations.extend([
                field_name.replace("HCFA", "").strip(),
                f"HCFA {field_name.split('HCFA')[-1].strip()}" if 'HCFA' in field_name else field_name,
                field_name.replace("HCFA ", ""),
                field_name.replace(" Code", ""),
            ])
        
        # Add common medical field variations
        if "diagnosis code" in field_name.lower():
            variations.extend([
                field_name.replace("Diagnosis Code", "ICD"),
                field_name.replace("Code", ""),
                f"ICD Diagnosis {field_name.split()[-1] if field_name.split() else ''}"
            ])
    
    elif field_name:
        # For field lookups without numbers
        variations.extend([
            field_name,
            field_name.upper(),
            field_name.lower(),
            field_name.title(),
            f"Technical Name: {field_name}",
            f"**{field_name}**",
            f"[{field_name}]",
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

# Enhanced Exact Match Detection with better patterns
def detect_exact_matches(state: RAGState) -> RAGState:
    """Enhanced exact match detection with better pattern matching"""
    search_variations = state["search_variations"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    exact_matches = []
    
    # Try multiple search strategies
    for search_term in search_variations[:8]:  # Increased from 5 to 8
        try:
            # Create embedding for search term
            query_embedding = embeddings_service.embed_query(search_term)
            
            # Search with high precision
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,  # Increased to get more candidates
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                    distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                    
                    # Enhanced exact match detection
                    is_exact = False
                    if extracted_number and extracted_field_name:
                        # Multiple exact match patterns
                        exact_patterns = [
                            rf"^{extracted_number}\.\s*{re.escape(extracted_field_name)}:",
                            rf"^\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}:",
                            rf"^{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]",
                            rf"^\*\*{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]",
                            rf"{extracted_number}\.\s*{re.escape(extracted_field_name)}[:\.]",
                            # Add underlined pattern
                            rf"\*\*{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]{{.underline}}:\*\*",
                        ]
                        
                        for pattern in exact_patterns:
                            if re.search(pattern, doc, re.IGNORECASE | re.MULTILINE):
                                is_exact = True
                                logger.info(f"Found exact match with pattern: {pattern}")
                                logger.info(f"Match: {extracted_number}. {extracted_field_name}")
                                break
                    
                    if is_exact or distance < 0.05:  # Very strict distance threshold for exact matches
                        exact_matches.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'is_exact_match': is_exact,
                            'search_term': search_term,
                            'match_quality': 'exact' if is_exact else 'very_close'
                        })
        
        except Exception as e:
            logger.error(f"Error in exact match detection: {e}")
    
    # Sort by exactness and distance
    exact_matches.sort(key=lambda x: (not x['is_exact_match'], x['distance']))
    
    # Log the best matches found
    if exact_matches:
        logger.info(f"Found {len(exact_matches)} exact/close matches")
        for i, match in enumerate(exact_matches[:3]):
            logger.info(f"  {i+1}. Distance: {match['distance']:.4f}, Exact: {match['is_exact_match']}")
            logger.info(f"      Preview: {match['content'][:100]}...")
    
    return {
        **state,
        "exact_matches": exact_matches[:10]  # Keep top 10 matches
    }

# Enhanced Numbered Definition Retrieval with better handling
def numbered_definition_retrieval(state: RAGState) -> RAGState:
    """Enhanced retrieval specifically for numbered definitions with better pattern matching"""
    search_variations = state["search_variations"]
    extracted_number = state["extracted_number"]
    extracted_field_name = state["extracted_field_name"]
    
    all_results = []
    seen_docs = set()
    
    # Use enhanced search variations
    for search_query in search_variations:
        try:
            query_embedding = embeddings_service.embed_query(search_query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,  # Increased for better coverage
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get('documents'):
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    if doc not in seen_docs:
                        seen_docs.add(doc)
                        metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                        distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                        
                        # Enhanced matching logic with more comprehensive patterns
                        is_exact_match = False
                        found_number = None
                        contains_definition = False
                        match_score = 0
                        
                        if extracted_number:
                            # Comprehensive exact match patterns
                            exact_patterns = [
                                (rf"^{extracted_number}\.\s*{re.escape(extracted_field_name)}:", 100),
                                (rf"^\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}:", 95),
                                (rf"^{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]", 90),
                                (rf"^\*\*{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]", 85),
                                (rf"{extracted_number}\.\s*{re.escape(extracted_field_name)}", 80),
                                # Add underlined patterns
                                (rf"\*\*{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]{{\.underline}}:\*\*", 98),
                                # Flexible patterns
                                (rf"{extracted_number}\.\s*.*{re.escape(extracted_field_name.split()[0])}", 70),
                            ]
                            
                            for pattern, score in exact_patterns:
                                if re.search(pattern, doc, re.IGNORECASE | re.MULTILINE):
                                    is_exact_match = True
                                    found_number = extracted_number
                                    match_score = score
                                    logger.info(f"Found exact match: {extracted_number}. {extracted_field_name} (score: {score})")
                                    break
                            
                            # Check for partial match with correct number
                            if not is_exact_match:
                                number_match = re.search(rf"(\d+)\.\s*.*{re.escape(extracted_field_name.split()[0])}", doc, re.IGNORECASE)
                                if number_match:
                                    found_number = number_match.group(1)
                                    match_score = 60
                        
                        # Check if contains complete definition elements
                        definition_indicators = ['Format:', 'Technical Name:', 'Length:', 'Position', 'Definition:']
                        found_indicators = sum(1 for indicator in definition_indicators if indicator in doc)
                        contains_definition = found_indicators >= 3
                        
                        # Additional scoring based on content quality
                        content_score = 0
                        if 'Format:' in doc: content_score += 10
                        if 'Technical Name:' in doc: content_score += 15
                        if 'Definition:' in doc: content_score += 20
                        if 'Length:' in doc: content_score += 10
                        if 'Position' in doc: content_score += 10
                        
                        total_score = match_score + content_score
                        
                        all_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'is_exact_match': is_exact_match,
                            'found_number': found_number,
                            'contains_definition': contains_definition,
                            'retrieval_strategy': 'numbered_definition',
                            'search_query': search_query,
                            'match_score': total_score,
                            'found_indicators': found_indicators
                        })
        
        except Exception as e:
            logger.error(f"Error in numbered definition search: {e}")
    
    # Enhanced sorting: exact matches first, then by total score, then by definition completeness
    all_results.sort(key=lambda x: (
        not x.get('is_exact_match', False),
        -x.get('match_score', 0),
        not x.get('contains_definition', False),
        x['distance'],
        -x.get('found_indicators', 0),
        -len(x['content'])
    ))
    
    # Log the best results
    if all_results:
        logger.info(f"Found {len(all_results)} numbered definition results")
        for i, result in enumerate(all_results[:5]):
            logger.info(f"  {i+1}. Score: {result.get('match_score', 0)}, "
                       f"Exact: {result.get('is_exact_match', False)}, "
                       f"Indicators: {result.get('found_indicators', 0)}, "
                       f"Distance: {result['distance']:.4f}")
    
    return {
        **state,
        "retrieved_docs": state.get("retrieved_docs", []) + all_results[:20],
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
                n_results=8,
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
        "retrieved_docs": state.get("retrieved_docs", []) + all_results[:15],
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
        
        # 1. Source-based scoring with higher weights for exact matches
        if doc.get('source') == 'exact_match':
            relevance_score += 200.0  # Increased from 150
            if doc.get('is_exact_match'):
                relevance_score += 100.0  # Additional bonus for true exact matches
        elif doc.get('source') == 'table_data':
            relevance_score += 50.0
        
        # 2. Enhanced exact match pattern scoring with more patterns
        if query_type == "numbered_definition" and extracted_number and extracted_field_name:
            # Comprehensive exact match patterns with different weights
            exact_patterns = [
                (rf"^{extracted_number}\.\s*{re.escape(extracted_field_name)}:", 250.0),
                (rf"^\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}:", 240.0),
                (rf"^{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]", 230.0),
                (rf"^\*\*{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]", 220.0),
                # Add underlined patterns
                (rf"\*\*{extracted_number}\.\s*\[{re.escape(extracted_field_name)}\]{{\.underline}}:\*\*", 245.0),
                # More flexible patterns
                (rf"{extracted_number}\.\s*{re.escape(extracted_field_name)}", 200.0),
                (rf"\*\*{extracted_number}\.\s*{re.escape(extracted_field_name)}", 180.0),
                # Very flexible pattern for partial matches
                (rf"{extracted_number}\.\s*.*{re.escape(extracted_field_name.split()[0])}", 150.0),
            ]
            
            for pattern, score in exact_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    relevance_score += score
                    logger.info(f"  *** EXACT PATTERN MATCH *** {extracted_number}. {extracted_field_name} (Score: {score})")
                    break
        
        # 3. Technical name exact match (very high value)
        if extracted_field_name:
            tech_name_patterns = [
                (rf"Technical Name:\s*{re.escape(extracted_field_name.replace(' ', '_').upper())}", 150.0),
                (rf"Technical Name:\s*{re.escape(extracted_field_name.upper())}", 140.0),
                (rf"Technical Name:\s*{re.escape(extracted_field_name)}", 130.0),
            ]
            
            for pattern, score in tech_name_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    relevance_score += score
                    break
        
        # 4. Complete definition structure scoring with higher weights
        definition_elements = ['Format:', 'Technical Name:', 'Length:', 'Position', 'Definition:']
        element_count = sum(1 for element in definition_elements if element in content)
        relevance_score += element_count * 20.0  # Increased from 15.0
        
        # Bonus for complete definitions
        if element_count >= 4:
            relevance_score += 50.0  # Increased from 30.0
        if element_count == 5:
            relevance_score += 25.0  # Additional bonus for all elements
        
        # 5. Enhanced metadata-based scoring
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
        
        # Metadata indicators with enhanced scoring
        if safe_bool_convert(metadata.get("is_numbered_definition")):
            relevance_score += 60.0  # Increased from 40.0
        if safe_bool_convert(metadata.get("is_complete_definition")):
            relevance_score += 50.0  # Increased from 35.0
        if safe_bool_convert(metadata.get("is_field_spec")):
            relevance_score += 30.0  # Increased from 25.0
        
        # Definition completeness score
        completeness = safe_int_convert(metadata.get("definition_completeness", 0))
        relevance_score += completeness * 10.0  # Increased from 8.0
        
        # 6. Field number matching with exact match bonus
        if extracted_number:
            meta_number = metadata.get("definition_number", "")
            if meta_number == extracted_number:
                relevance_score += 80.0  # Increased from 60.0
        
        # 7. Content quality indicators
        content_length = len(content)
        if content_length > 200:  # Substantial content
            relevance_score += 15.0
        if content_length > 500:  # Very detailed content
            relevance_score += 25.0
        if content_length > 1000:  # Comprehensive content
            relevance_score += 15.0
        
        # 8. Distance-based scoring (inverse relationship) - enhanced
        distance_score = max(0, 40.0 * (1.0 - distance))  # Increased from 30.0
        relevance_score += distance_score
        
        # 9. Search strategy bonus with enhanced weights
        if doc.get('is_exact_match'):
            relevance_score += 100.0  # Increased from 80.0
        if doc.get('is_tech_name_match'):
            relevance_score += 70.0  # Increased from 60.0
        if doc.get('is_field_match'):
            relevance_score += 60.0  # Increased from 50.0
        if doc.get('match_score'):
            relevance_score += doc['match_score'] * 0.5  # Use the match score from numbered def retrieval
        
        # 10. Content type specific scoring
        content_type = metadata.get('content_type', '')
        if content_type == 'numbered_definition':
            relevance_score += 40.0  # Increased from 25.0
        elif content_type == 'field_specification':
            relevance_score += 30.0  # Increased from 20.0
        elif content_type == 'value_mapping':
            relevance_score += 20.0  # Increased from 15.0
        
        # 11. Table and appendix bonuses
        if safe_bool_convert(metadata.get("has_embedded_tables")):
            relevance_score += 20.0
        if content_type == 'appendix':
            relevance_score += 15.0
        
        # 12. Enhanced penalty for incomplete or very short content
        if content_length < 50:
            relevance_score -= 25.0  # Increased penalty
        elif content_length < 100:
            relevance_score -= 10.0
        
        # 13. Bonus for fields containing specific indicators
        if extracted_field_name and extracted_field_name.lower() in content.lower():
            relevance_score += 30.0
        
        # Calculate confidence score
        confidence = min(1.0, relevance_score / 300.0)  # Increased denominator to account for higher scores
        
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
                'contains_definition': 'Definition:' in content,
                'contains_number': extracted_number in content if extracted_number else False,
                'contains_field_name': extracted_field_name.lower() in content.lower() if extracted_field_name else False
            }
        })
    
    # Sort by relevance score (descending)
    processed_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Calculate overall confidence based on top result
    top_score = processed_docs[0]['relevance_score'] if processed_docs else 0
    confidence_score = min(1.0, top_score / 300.0)
    
    # Enhanced logging with more details
    logger.info("\n--- Enhanced Document Ranking Results ---")
    for i, doc in enumerate(processed_docs[:5]):
        metadata = doc['metadata']
        logger.info(f"{i+1}. Score: {doc['relevance_score']:.1f}, Distance: {doc['distance']:.3f}, Confidence: {doc['confidence']:.2f}")
        logger.info(f"   Strategy: {doc.get('retrieval_strategy', 'unknown')}, Source: {doc.get('source', 'unknown')}")
        logger.info(f"   Content preview: {doc['content'][:150]}...")
        
        # Check if content contains the field number we're looking for
        content = doc['content']
        if extracted_number:
            if f"{extracted_number}." in content:
                logger.info(f"   *** Contains field {extracted_number} ***")
                # Extract a snippet around the field number
                pattern = rf"({extracted_number}\..*?)(?=\n\d+\.|$)"
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    snippet = match.group(1)[:300] + "..." if len(match.group(1)) > 300 else match.group(1)
                    logger.info(f"   Snippet: {snippet}")
        
        # Log quality indicators
        quality = doc.get('quality_indicators', {})
        logger.info(f"   Quality: Elements={quality.get('definition_elements', 0)}, "
                   f"HasNum={quality.get('contains_number', False)}, "
                   f"HasName={quality.get('contains_field_name', False)}")
    
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
    
    # Enhanced filtering with more flexible criteria
    high_quality_docs = [
        doc for doc in processed_docs 
        if doc['distance'] <= RETRIEVAL_DISTANCE_THRESHOLD and doc['relevance_score'] > 50.0
    ]
    
    if not high_quality_docs:
        # Relax criteria if no high-quality docs found
        high_quality_docs = [
            doc for doc in processed_docs 
            if doc['distance'] <= 0.95 and doc['relevance_score'] > 25.0
        ][:5]
    
    if not high_quality_docs:
        # Further relaxation for any decent matches
        high_quality_docs = processed_docs[:3]
    
    # For numbered definitions, prioritize exact matches and complete definitions
    if query_type == "numbered_definition" and extracted_number:
        # Separate exact matches from others
        exact_matches = [doc for doc in high_quality_docs if doc.get('relevance_score', 0) > 200]
        other_docs = [doc for doc in high_quality_docs if doc.get('relevance_score', 0) <= 200]
        
        # Prioritize exact matches
        final_docs = exact_matches[:3] + other_docs[:2]
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

# Enhanced Answer Generation Node with much better prompting
def generate_answer(state: RAGState) -> RAGState:
    """Generate properly formatted answer with enhanced prompting"""
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
    
    # Enhanced prompt for numbered definitions with much better instructions
    if query_type == "numbered_definition" and extracted_number:
        prompt_template = """You are an expert medical data dictionary assistant. Your task is to extract and format field definitions exactly as they appear in the source document.

The user asked about: "{original_question}"
Looking for field number: {field_number}
Field name: "{field_name}"

CRITICAL INSTRUCTIONS FOR FORMATTING:
1. Find the complete numbered definition that starts with "{field_number}. {field_name}" in the context
2. Extract ALL field details and format them EXACTLY as shown below
3. Use this EXACT format (copy this structure exactly):

**{field_number}. {field_name}:**
• **Format:** [extract the exact format value from the context]
• **Technical Name:** [extract the exact technical name from the context]
• **Length:** [extract the exact length value from the context]
• **Positions:** [extract the exact positions value from the context]
• **Definition:** [extract the complete definition text from the context]

IMPORTANT EXTRACTION RULES:
- Find lines that say "Format: [value]" and extract the value after the colon
- Find lines that say "Technical Name: [value]" and extract the value after the colon
- Find lines that say "Length: [value]" and extract the value after the colon
- Find lines that say "Position: [value]" or "Positions: [value]" and extract the value after the colon
- Find lines that say "Definition: [value]" and extract the complete definition text
- If any field is missing, write "Not specified" instead
- Include any NOTE: sections that appear after the main definition
- Preserve all original formatting, spacing, and punctuation

EXAMPLE OF CORRECT OUTPUT:
**141. HCFA Admit Type Code:**
• **Format:** Character
• **Technical Name:** HCFA_ADMIT_TYPE_CD
• **Length:** 1
• **Positions:** 1236
• **Definition:** A value which represents a classification of a member admission to a facility as it would be represented on a standardized UB92 claim form. Please refer to the Appendices for a listing of the specific values and their definitions.

Context from the medical data dictionary:
---
{context}
---

Extract and format the numbered definition using the EXACT format above:"""
        
        final_prompt = prompt_template.format(
            original_question=query,
            field_number=extracted_number,
            field_name=extracted_field_name,
            context=context_string
        )
        
    elif query_type == "field_lookup":
        prompt_template = """You are an expert medical data dictionary assistant. Extract field information from the context.

The user asked about: "{original_question}"
Looking for field: "{field_name}"

INSTRUCTIONS:
1. Find the complete definition for this field in the context
2. If it's a numbered definition, use this exact format:

**[NUMBER]. [FIELD NAME]:**
• **Format:** [value]
• **Technical Name:** [value]
• **Length:** [value]
• **Positions:** [value]
• **Definition:** [complete definition]

3. If it's not numbered, present all available information clearly
4. Include any embedded tables, value mappings, or related information
5. Include any notes or special information that appears after the definition
6. Extract values exactly as they appear in the context

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
        prompt_template = """You are an expert medical data dictionary assistant. Explain concepts from the medical data dictionary.

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
6. Include any notes or appendices referenced

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
        
        # Enhanced post-processing
        final_answer = post_process_answer(final_answer, query_type, extracted_number, extracted_field_name)
        
        # Adjust confidence based on answer quality
        if "could not find" in final_answer.lower() or "no information" in final_answer.lower():
            confidence_score *= 0.3
        elif extracted_number and f"**{extracted_number}." in final_answer:
            confidence_score = min(1.0, confidence_score + 0.2)
        elif extracted_field_name and extracted_field_name in final_answer:
            confidence_score = min(1.0, confidence_score + 0.1)
        
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

def post_process_answer(answer: str, query_type: str, extracted_number: Optional[str], extracted_field_name: Optional[str]) -> str:
    """Enhanced post-processing of the answer to ensure proper formatting"""
    # Ensure proper bullet point formatting
    answer = re.sub(r'^[\s-]*\*\s*\*\*([^:]+):\*\*', r'• **\1:**', answer, flags=re.MULTILINE)
    answer = re.sub(r'^[\s-]*\*\s*([^:]*:)', r'• **\1**', answer, flags=re.MULTILINE)
    
    # Ensure proper numbered definition header
    if query_type == "numbered_definition" and extracted_number and extracted_field_name:
        # Make sure the header is properly formatted
        header_patterns = [
            (rf'^{extracted_number}\.([^:]+):', rf'**{extracted_number}.\1:**'),
            (rf'^\*\*{extracted_number}\.([^:*]+):', rf'**{extracted_number}.\1:**'),
            (rf'^{extracted_number}\.\s*{re.escape(extracted_field_name)}', rf'**{extracted_number}. {extracted_field_name}:**'),
        ]
        
        for pattern, replacement in header_patterns:
            answer = re.sub(pattern, replacement, answer, flags=re.MULTILINE)
        
        # Ensure the field name is in the header if missing
        if f"**{extracted_number}." in answer and extracted_field_name not in answer:
            answer = re.sub(rf'\*\*{extracted_number}\.\s*([^:]*?):\*\*', 
                           f'**{extracted_number}. {extracted_field_name}:**', answer)
    
    # Clean up extra whitespace
    answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
    
    # Ensure bullet points are properly spaced
    answer = re.sub(r'(\*\*[^:]+:\*\*)\n([^•])', r'\1\n\n\2', answer)
    
    return answer.strip()

# Validation Node
def validate_answer(state: RAGState) -> RAGState:
    """Enhanced validation of the generated answer"""
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
        
        # Check if all expected fields are present
        if 'Format:' in final_answer and 'Technical Name:' in final_answer and 'Definition:' in final_answer:
            validation_score += 0.15
        
        # Bonus for exact match in format
        if f"**{extracted_number}. {extracted_field_name}:**" in final_answer:
            validation_score += 0.25
    
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
    if 'Definition:' in final_answer:  # Contains definition
        validation_score += 0.1
    
    # Cap the validation score
    validation_score = min(1.0, validation_score)
    
    # Log validation details
    logger.info(f"Validation score: {validation_score:.3f} (original: {confidence_score:.3f})")
    
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