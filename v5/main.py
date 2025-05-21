# Enhanced RAG system with Advanced LangGraph workflow
# Optimized for numbered definitions, embedded tables, and medical data dictionary

import os
import sys
import logging
import re
import json
from typing import TypedDict, List, Dict, Optional, Literal, Union, Any
from pathlib import Path
from dotenv import load_dotenv 
from flask import Flask, request, jsonify, render_template 
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
import chromadb

# Set up API key for Google Cloud authentication
API_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.json")
if os.path.exists(API_KEY_PATH):
    # Set the environment variable to use this key file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_PATH
    print(f"Using Google credentials from: {API_KEY_PATH}")
    
    # Try to extract project ID from the key file if not set in environment
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        try:
            with open(API_KEY_PATH, 'r') as key_file:
                key_data = json.load(key_file)
                if 'project_id' in key_data:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = key_data['project_id']
                    print(f"Using project ID from key file: {key_data['project_id']}")
        except Exception as e:
            print(f"Warning: Could not extract project ID from key file: {e}")
else:
    print(f"Warning: API key file not found at {API_KEY_PATH}")

# Load environment variables from .env file
# Look for .env files in multiple locations
env_paths = [
    '.env',                      # Same directory
    '../.env',                   # Parent directory
    os.path.expanduser('~/.env') # Home directory
]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        env_loaded = True
        print(f"Loaded environment variables from {env_path}")
        break

if not env_loaded:
    print("Warning: No .env file found. Using environment variables as is.")

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

# --- Configuration ---
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-005")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gemini-2.5-pro-preview-05-06")  # Default model
AVAILABLE_MODELS = {
    "gemini-2.5-pro-preview-05-06": {"temperature": 0.02},
    "gemini-2.5-flash-preview-04-17": {"temperature": 0.05}
}
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aethrag2")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.02"))
NUM_RETRIEVED_DOCS = int(os.getenv("NUM_RETRIEVED_DOCS", "30"))
RETRIEVAL_DISTANCE_THRESHOLD = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.88"))
FIELD_DEFINITIONS_FILE = os.getenv("FIELD_DEFINITIONS_FILE", "field_definitions.json")

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Print important environment variables for debugging
logger.info(f"GOOGLE_CLOUD_PROJECT: {GCP_PROJECT_ID}")
logger.info(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
logger.info(f"CHROMA_DB_DIR: {CHROMA_DB_DIR}")
logger.info(f"CHAT_MODEL_NAME: {CHAT_MODEL_NAME}")

app = Flask(__name__, template_folder='templates')

# Format configuration - set to True for line breaks, False for dash format
USE_LINE_BREAKS = os.getenv("USE_LINE_BREAKS", "True").lower() in ("true", "1", "yes")

# Dictionary of known field definitions - a starter set of common fields
FIELD_DEFINITIONS = {
    "77": {
        "title": "Place of Service",
        "format": "Character",
        "technical_name": "POS_CD",
        "length": "1",
        "positions": "1049",
        "definition": "A code that identifies where the service was performed. I = Inpatient, O = Outpatient, H = Home, P = Provider Office, L = Long Term Care/Skilled Nursing Facility, T = Clinic, X = Ambulatory Surgical Center."
    },
    "99": {
        "title": "Benefit Payable",
        "format": "Numeric",
        "technical_name": "BNFT_PAYABLE_AMT",
        "length": "10",
        "positions": "1115-1124",
        "definition": "The dollar amount the provider received for the services on this claim (or, for pharmacy claims, the dollar amount the pharmacy received)."
    },
    "100": {
        "title": "Paid Amount",
        "format": "Numeric",
        "technical_name": "PAID_AMT",
        "length": "10",
        "positions": "1125-1134",
        "definition": "The dollar amount the plan paid for the services received."
    },
    "101": {
        "title": "Coordination of Benefits Paid Amount",
        "format": "Numeric",
        "technical_name": "COB_PAID_AMT",
        "length": "10",
        "positions": "1135-1144",
        "definition": "The dollar amount paid by another insurance carrier, Medicare, or other third party. This amount is applied to reduce the payer's liability. This field will be populated with values greater than zero only if patient has other insurance and the other carrier(s) paid a portion of the claim."
    },
    "123": {
        "title": "Billed Eligible Amount",
        "format": "Numeric",
        "technical_name": "BILLED_ELIGIBLE_AMT",
        "length": "10",
        "positions": "1175-1184",
        "definition": "The total dollar amount that is considered eligible for consideration in the claims adjudication process.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero on specific records deemed to contain sensitive information."
    },
    "139": {
        "title": "HCFA Place of Service Code",
        "format": "Character",
        "technical_name": "HCFA_POS_CD",
        "length": "2",
        "positions": "1234-1235",
        "definition": "A two digit standardized code which identifies where a service was rendered. Values are based on the CMS (formerly HCFA) Place of Service codes."
    },
    "140": {
        "title": "HCFA Admit Source Code",
        "format": "Character",
        "technical_name": "HCFA_ADMIT_SOURCE_CD",
        "length": "1",
        "positions": "1235",
        "definition": "A value which represents a source of admission to a facility as it would be represented on a standardized UB92 claim form. Please refer to the Appendices for a listing of the specific values and their definitions."
    },
    "141": {
        "title": "HCFA Admit Type Code",
        "format": "Character",
        "technical_name": "HCFA_ADMIT_TYPE_CD",
        "length": "1",
        "positions": "1236",
        "definition": "A value which represents a classification of a member admission to a facility as it would be represented on a standardized UB92 claim form. Please refer to the Appendices for a listing of the specific values and their definitions."
    },
    "153": {
        "title": "Diagnosis Code 1",
        "format": "Character",
        "technical_name": "ICD_DX_CD_1",
        "length": "7",
        "positions": "1357-1363",
        "definition": "The first ICD-9 or ICD-10 Diagnosis Code associated with the services on a given claim record.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    },
    "154": {
        "title": "Diagnosis Code 2",
        "format": "Character",
        "technical_name": "ICD_DX_CD_2",
        "length": "7",
        "positions": "1364-1370",
        "definition": "The second ICD-9 or ICD-10 Diagnosis Code associated with the services on a given claim record.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    },
    "155": {
        "title": "Diagnosis Code 3",
        "format": "Character",
        "technical_name": "ICD_DX_CD_3",
        "length": "7",
        "positions": "1371-1377",
        "definition": "The third ICD-9 or ICD-10 Diagnosis Code associated with the services on a given claim record.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    },
    "156": {
        "title": "Diagnosis Code 4",
        "format": "Character",
        "technical_name": "ICD_DX_CD_4",
        "length": "7",
        "positions": "1378-1384",
        "definition": "The fourth ICD-9 or ICD-10 Diagnosis Code associated with the services on a given claim record.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    },
    "157": {
        "title": "Diagnosis Code 5",
        "format": "Character",
        "technical_name": "ICD_DX_CD_5",
        "length": "7",
        "positions": "1385-1391",
        "definition": "The fifth ICD-9 or ICD-10 Diagnosis Code associated with the services on a given claim record.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    },
    "158": {
        "title": "Diagnosis Code 6",
        "format": "Character",
        "technical_name": "ICD_DX_CD_6",
        "length": "7",
        "positions": "1392-1398",
        "definition": "The sixth ICD-9 or ICD-10 Diagnosis Code associated with the services on a given claim record.\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    },
    "163": {
        "title": "Claim-Level ICD Procedure Code 1",
        "format": "Character",
        "technical_name": "ICD_PROC_CD_1",
        "length": "7",
        "positions": "1417-1423",
        "definition": "A value in the ICD-9 or ICD-10 medical coding system, identifying an operating room procedure as it is was recorded by a hospital facility on a standardized UB92 inpatient billing claim form. This field contains the primary ICD-9 or ICD-10 procedure code, as identified by the hospital. This field is populated only for inpatient facility claims (Type of Service (Field #74 = 50 ) and Place of Service (Field #77 = I )).\nNote: We have noticed that some hospitals, when electronically submitting their claims, incorrectly enter a HCPCS code instead of an ICD-9 or ICD-10 procedure code in this field. If you find what appears to be an invalid ICD-9 or ICD-10 procedure code, and the first byte of that code is a \"letter\" value, the code is probably a truncated HCPCS code. For example, if the hospital enters the HCPCS code E1390 in this field, the field formatting will display that code on this record as E13.9 (the decimal is inserted based on the expectation that the code would be an ICD-9 procedure code).\nNOTE: For split-funded or full-risk members, this field may show as a value of zero/blank on specific records deemed to contain sensitive information."
    }
    # Add more field definitions as needed
}

# Function to load field definitions from a JSON file
def load_field_definitions(file_path=FIELD_DEFINITIONS_FILE):
    """Load field definitions from a JSON file and merge with existing definitions"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                import json
                loaded_definitions = json.load(file)
                # Merge with existing definitions, loaded definitions take precedence
                FIELD_DEFINITIONS.update(loaded_definitions)
                logger.info(f"Loaded {len(loaded_definitions)} field definitions from {file_path}")
    except Exception as e:
        logger.error(f"Error loading field definitions from {file_path}: {e}")

# Function to save field definitions to a JSON file
def save_field_definitions(file_path=FIELD_DEFINITIONS_FILE):
    """Save current field definitions to a JSON file"""
    try:
        import json
        with open(file_path, 'w') as file:
            json.dump(FIELD_DEFINITIONS, file, indent=2)
            logger.info(f"Saved {len(FIELD_DEFINITIONS)} field definitions to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving field definitions to {file_path}: {e}")
        return False

# Helper function to format field definitions
def format_field_definition(field_id: str, with_line_breaks: bool = True) -> str:
    """Format a field definition with either line breaks or dashes"""
    field_data = FIELD_DEFINITIONS.get(field_id)
    if not field_data:
        return None
        
    title = field_data.get("title", f"Field {field_id}")
    fmt = field_data.get("format", "Character")
    tech_name = field_data.get("technical_name", "")
    length = field_data.get("length", "")
    positions = field_data.get("positions", "")
    definition = field_data.get("definition", "No definition available")
    
    # Ensure we're not displaying "Unknown" or empty values
    if tech_name == "Unknown" or not tech_name:
        # Try to derive a technical name from the title if needed
        if title:
            tech_name = title.replace(" ", "_").upper()
        else:
            tech_name = f"FIELD_{field_id}"
    
    # Ensure we have non-empty values for all fields
    fmt = fmt if fmt and fmt != "Unknown" else "Character"
    length = length if length and length != "Unknown" else "N/A"
    positions = positions if positions and positions != "Unknown" else "N/A"
    
    # Remove any trailing colons from title
    title = title.rstrip(':')
    
    if with_line_breaks:
        return f"""**{field_id}. {title}:**
**Format:** {fmt}
**Technical Name:** {tech_name}
**Length:** {length}
**Positions:** {positions}
**Definition:** {definition}"""
    else:
        return f"**{field_id}. {title}:** - **Format:** {fmt} - **Technical Name:** {tech_name} - **Length:** {length} - **Positions:** {positions} - **Definition:** {definition}"

# Try to load external field definitions when module is imported
try:
    load_field_definitions()
except Exception as e:
    logger.warning(f"Warning: Could not load external field definitions: {e}")

# Check if a field ID exists in our definitions
def has_field_definition(field_id: str) -> bool:
    """Check if we have a definition for this field ID"""
    return field_id in FIELD_DEFINITIONS

# Check if field name matches a defined field
def get_field_id_by_name(field_name: str) -> Optional[str]:
    """Try to match a field name to our definitions"""
    if not field_name:
        return None
        
    # Clean up the field name - remove common prefixes like "explain" and trim whitespace
    field_name = field_name.lower().strip()
    prefixes_to_remove = ["explain ", "what is ", "tell me about ", "describe "]
    for prefix in prefixes_to_remove:
        if field_name.startswith(prefix):
            field_name = field_name[len(prefix):].strip()
            break
    
    # Direct mapping for known field names
    if field_name in FIELD_NAME_TO_NUMBER:
        return FIELD_NAME_TO_NUMBER[field_name]
    
    # Try without trailing punctuation
    if field_name.endswith((':', '?', '.')):
        clean_name = field_name[:-1].strip()
        if clean_name in FIELD_NAME_TO_NUMBER:
            return FIELD_NAME_TO_NUMBER[clean_name]
    
    # Search through definitions for matching title (case insensitive)
    for field_id, field_data in FIELD_DEFINITIONS.items():
        title = field_data.get("title", "").lower()
        if title and (field_name == title or field_name == title.rstrip(':')):
            return field_id
        
        # Also try matching just the primary part of the title (without qualifiers)
        if title and title.split(':', 1)[0].strip() == field_name:
            return field_id
    
    # Fuzzy matching - look for partial matches
    # Sort potential matches by length of match (longest first)
    potential_matches = []
    for field_id, field_data in FIELD_DEFINITIONS.items():
        title = field_data.get("title", "").lower().rstrip(':')
        if title and field_name in title:
            match_length = len(field_name)
            potential_matches.append((field_id, match_length, title))
    
    # Sort by match length (descending)
    if potential_matches:
        potential_matches.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Fuzzy matched '{field_name}' to field {potential_matches[0][0]} ({potential_matches[0][2]})")
        return potential_matches[0][0]
            
    return None

# Add this function after the configuration section and before init_clients_rag
def verify_google_auth():
    """
    Verify Google API authentication status and provide helpful feedback
    """
    auth_status = {
        "credentials_found": False,
        "project_id_found": False,
        "credentials_path": None,
        "project_id": None,
        "issues": []
    }
    
    # Check for credentials file
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        auth_status["credentials_path"] = credentials_path
        if os.path.exists(credentials_path):
            auth_status["credentials_found"] = True
            logger.info(f"Found credentials at: {credentials_path}")
        else:
            auth_status["issues"].append(f"Credentials file not found at: {credentials_path}")
            logger.warning(f"Credentials file not found at: {credentials_path}")
    elif os.path.exists(API_KEY_PATH):
        auth_status["credentials_path"] = API_KEY_PATH
        auth_status["credentials_found"] = True
        logger.info(f"Using API key file: {API_KEY_PATH}")
    else:
        auth_status["issues"].append("No credentials file found")
        logger.warning("No credentials file found")
    
    # Check for project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        auth_status["project_id"] = project_id
        auth_status["project_id_found"] = True
    else:
        auth_status["issues"].append("GOOGLE_CLOUD_PROJECT environment variable not set")
        logger.warning("GOOGLE_CLOUD_PROJECT environment variable not set")
    
    # Try to validate credentials if found
    if auth_status["credentials_found"] and auth_status["project_id_found"]:
        try:
            # Simple connection test to verify credentials work
            from google.cloud import storage
            storage_client = storage.Client()
            # Just calling a simple method to test authentication
            _ = list(storage_client.list_buckets(max_results=1))
            logger.info("Successfully authenticated with Google Cloud")
        except Exception as e:
            auth_status["issues"].append(f"Authentication test failed: {str(e)}")
            logger.warning(f"Authentication test failed: {e}")
    
    return auth_status

# Call this before initializing the clients
try:
    auth_status = verify_google_auth()
    if auth_status["issues"]:
        logger.warning(f"Authentication issues: {', '.join(auth_status['issues'])}")
    else:
        logger.info("Google Cloud authentication verified successfully")
except Exception as e:
    logger.error(f"Error verifying authentication: {e}")

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
    "diagnosis code 2": "154",
    "diagnosis code 3": "155",
    "diagnosis code 4": "156",
    "diagnosis code 5": "157",
    "diagnosis code 7": "159",
    "diagnosis code 8": "160",
    "diagnosis code 9": "161",
    "diagnosis code 10": "162",
    "ahf_bfd_amt": "102",
    "aetna health fund before fund deductible": "102",
    "ahf before fund deductible": "102",
    "cob paid amount": "101",
    "coordination of benefits paid amount": "101",
    "paid amount": "100",
    "benefit payable": "99",
    "billed eligible amount": "123",
    "claim-level icd procedure code 1": "163",
    "claim level icd procedure code 1": "163",
    "icd procedure code 1": "163",
    "claim-level procedure code 1": "163",
    "procedure code 1": "163",
    "claim level procedure code 1": "163",
    "claim-level icd procedure code 2": "164",
    "claim-level icd procedure code 3": "165",
    "claim-level icd procedure code 4": "166",
    "claim-level icd procedure code 5": "167",
    "claim-level icd procedure code 6": "168",
    "line-level procedure code": "68",
    "icd procedure code 1": "163",
    "type of service": "74",
    "admission date": "142",
    "discharge date": "143",
    "aetna health fund determination order code": "169",
    "aetna health fund member copay amount": "171",
    "member copay amount": "171",
    "member deductible amount": "172",
    "aetna health fund member deductible amount": "172"
}

def init_clients_rag(): 
    global embeddings_service, chat_model, chroma_client, collection
    
    # Verify GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        if os.path.exists(API_KEY_PATH):
            logger.info(f"Using detected API key file: {API_KEY_PATH}")
        else:
            logger.warning("No API key file found. Authentication may fail.")
            
    if not GCP_PROJECT_ID:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is essential and not set.")
    
    # Initialize Vertex AI embedding service
    if not embeddings_service:
        try:
            embeddings_service = VertexAIEmbeddings(
                model_name=EMBEDDING_MODEL_NAME, 
                project=GCP_PROJECT_ID
            )
            logger.info(f"Successfully initialized VertexAI Embeddings with model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize VertexAI Embeddings: {e}")
            raise ValueError(f"Error initializing embeddings service: {e}")
    
    # Initialize Chroma client
    if not chroma_client:
        if not os.path.exists(CHROMA_DB_DIR):
            raise FileNotFoundError(f"ChromaDB directory '{CHROMA_DB_DIR}' missing. Run create_chroma_db.py first.")
        
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            logger.info(f"Connected to ChromaDB at: {CHROMA_DB_DIR}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    # Get the collection
    if not collection:
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            logger.info(f"Loaded Chroma collection '{COLLECTION_NAME}' ({collection.count()} items).")
        except Exception as e:
            logger.error(f"Error: Collection '{COLLECTION_NAME}' not found: {e}")
            raise

def get_chat_model(model_name=None):
    """
    Initialize and return a ChatVertexAI model with the specified parameters
    or default parameters if none are provided.
    """
    if not model_name or model_name not in AVAILABLE_MODELS:
        model_name = CHAT_MODEL_NAME  # Default model
    
    model_config = AVAILABLE_MODELS[model_name]
    temperature = model_config.get("temperature", LLM_TEMPERATURE)
    
    # Check for required auth
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.path.exists(API_KEY_PATH):
        logger.warning("No API key file or GOOGLE_APPLICATION_CREDENTIALS found. Authentication may fail.")
        
    if not GCP_PROJECT_ID:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set.")
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is essential and not set.")
    
    try:
        logger.info(f"Initializing ChatVertexAI with model: {model_name}, temperature: {temperature}")
        return ChatVertexAI(
            model_name=model_name, 
            project=GCP_PROJECT_ID, 
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatVertexAI model: {e}")
        raise ValueError(f"Error initializing ChatVertexAI model: {e}")

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
            r"(?:what\s+is\s+|define\s+|explain\s+)?([A-Za-z][A-Za-z0-9\s\-_]+(?:Code|Amount|ID|Number|Date|Type|Name|Status|Category))\s*\??$",
            r"^(?:explain|describe|tell\s+me\s+about)\s+([A-Za-z][A-Za-z0-9\s\-_]+)$"  # Explicit explain pattern
        ]
        
        for pattern in field_name_patterns:
            match = re.search(pattern, query.strip(), re.IGNORECASE)
            if match:
                query_type = "field_lookup"
                extracted_field_name = match.group(1).strip()
                
                # Check if we have a field mapping
                # Use our enhanced get_field_id_by_name function that handles "explain" prefixes
                field_id = get_field_id_by_name(extracted_field_name)
                if field_id:
                    extracted_number = field_id
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
    
    # Check if we have this field in our definitions first
    if extracted_number and has_field_definition(extracted_number):
        formatted_answer = format_field_definition(extracted_number, USE_LINE_BREAKS)
        return {
            **state,
            "final_answer": formatted_answer,
            "confidence_score": 0.9  # High confidence for known fields
        }
    
    # Check if we can match the field name to a known field ID
    if extracted_field_name:
        field_id = get_field_id_by_name(extracted_field_name)
        if field_id and has_field_definition(field_id):
            formatted_answer = format_field_definition(field_id, USE_LINE_BREAKS)
            return {
                **state,
                "final_answer": formatted_answer,
                "confidence_score": 0.9  # High confidence for known fields
            }
    
    # If no match in our definitions, proceed with standard processing
    
    # Create specialized prompt for numbered definitions
    if query_type == "numbered_definition" and extracted_number:
        prompt_template = """You are an AI assistant that extracts numbered field definitions from a medical data dictionary.

The user asked about: "{original_question}"
Looking for field number: {field_number}
Field name: "{field_name}"

CRITICAL INSTRUCTIONS:
1. Find the COMPLETE numbered definition that starts with "{field_number}. {field_name}" in the context
2. IMPORTANT: You MUST extract values for ALL fields below. Do not leave any field blank.
3. If you cannot find a specific value, use "N/A" instead of leaving it blank
4. The output must follow this EXACT format with ACTUAL VALUES:

**{field_number}. {field_name}:**
**Format:** [Actual format value, e.g. Character, Numeric]
**Technical Name:** [Actual technical name, e.g. HCFA_ADMIT_TYPE_CD]
**Length:** [Actual length value, e.g. 1, 2]
**Positions:** [Actual positions value, e.g. 1236]
**Definition:** [Actual definition text]

5. For each field, look for patterns like:
   - Format: Usually "Character", "Numeric", "Date", etc.
   - Technical Name: Often uppercase with underscores like "HCFA_ADMIT_TYPE_CD"
   - Length: A number representing character count like "1", "2", etc.
   - Positions: A number representing the position in a record like "1236"
   - Definition: The full explanation of the field's purpose

6. Include any additional notes, especially those starting with "NOTE:" after the Definition
7. If you cannot find the exact field, try to find similar fields or related information

Context from the medical data dictionary:
---
{context}
---

Extract and format the numbered definition with COMPLETE VALUES for all fields:"""
        
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

CRITICAL INSTRUCTIONS:
1. Find the complete definition for this field
2. IMPORTANT: You MUST extract values for ALL fields below. Do not leave any field blank.
3. If you cannot find a specific value, use "N/A" instead of leaving it blank
4. If it's a numbered definition, use this EXACT format with ACTUAL VALUES:

**[NUMBER]. [FIELD NAME]:**
**Format:** [Actual format value]
**Technical Name:** [Actual technical name]
**Length:** [Actual length value]
**Positions:** [Actual positions value]
**Definition:** [Actual definition text]

5. For each field, look for patterns like:
   - Format: Usually "Character", "Numeric", "Date", etc.
   - Technical Name: Often uppercase with underscores like "HCFA_ADMIT_TYPE_CD"
   - Length: A number representing character count like "1", "2", etc.
   - Positions: A number representing the position in a record like "1236"
   - Definition: The full explanation of the field's purpose

6. Include any embedded tables, value mappings, or related information
7. If you cannot find the exact field, try to find similar fields or related information

Context from the medical data dictionary:
---
{context}
---

Provide the complete field information with ALL VALUES filled in:"""
        
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
2. IMPORTANT: If the context contains numbered definitions, you MUST include values for ALL fields below. Do not leave any field blank.
3. If you cannot find a specific value, use "N/A" instead of leaving it blank
4. If you find a numbered definition, use this EXACT format with ACTUAL VALUES:

**[NUMBER]. [FIELD NAME]:**
**Format:** [Actual format value]
**Technical Name:** [Actual technical name]
**Length:** [Actual length value]
**Positions:** [Actual positions value]
**Definition:** [Actual definition text]

5. For each field, look for patterns like:
   - Format: Usually "Character", "Numeric", "Date", etc.
   - Technical Name: Often uppercase with underscores like "HCFA_ADMIT_TYPE_CD"
   - Length: A number representing character count like "1", "2", etc.
   - Positions: A number representing the position in a record like "1236"
   - Definition: The full explanation of the field's purpose

6. Include any relevant tables, value mappings, or cross-references
7. If you cannot find the exact information, try to find similar concepts or related information

Context from the medical data dictionary:
---
{context}
---

Provide a comprehensive explanation with ALL VALUES filled in for any field definitions:"""
        
        final_prompt = prompt_template.format(
            original_question=query,
            topic=user_query_term,
            context=context_string
        )
    
    # Generate the answer
    try:
        response = chat_model.invoke(final_prompt)
        raw_answer = response.content.strip()
        
        try:
            # Try regular post-processing
            final_answer = post_process_answer(raw_answer, query_type, extracted_number, extracted_field_name)
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            # Fall back to simplified formatting
            final_answer = simplified_post_process(raw_answer, query_type, extracted_number, extracted_field_name)
        
        # Adjust confidence based on answer quality
        if "could not find" in final_answer.lower() or "no information" in final_answer.lower():
            confidence_score *= 0.3
        elif extracted_number and f"{extracted_number}." in final_answer:
            confidence_score = min(1.0, confidence_score + 0.2)
        
        # Check for missing values and adjust confidence
        missing_value_pattern = r'\*\*[^:]+:\*\*\s+(N/A|-|\n)' if USE_LINE_BREAKS else r'\*\*[^:]+:\*\*\s+(-|\*\*)'
        if re.search(missing_value_pattern, final_answer):
            # Found empty field values
            confidence_score *= 0.5
            logger.warning("Found empty field values in response")
        
        return {
            **state,
            "final_answer": final_answer,
            "confidence_score": confidence_score
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        
        # Try to use field definitions as fallback
        if extracted_number and has_field_definition(extracted_number):
            formatted_answer = format_field_definition(extracted_number, USE_LINE_BREAKS)
            return {
                **state,
                "final_answer": formatted_answer,
                "confidence_score": 0.85  # Good confidence for fallback
            }
        
        # Try to find field by name
        if extracted_field_name:
            field_id = get_field_id_by_name(extracted_field_name)
            if field_id and has_field_definition(field_id):
                formatted_answer = format_field_definition(field_id, USE_LINE_BREAKS)
                return {
                    **state,
                    "final_answer": formatted_answer,
                    "confidence_score": 0.85  # Good confidence for fallback
                }
        
        # No fallback available
        return {
            **state,
            "final_answer": f"Error generating response. Please try a different query or provide more specific details.",
            "confidence_score": 0.0
        }

def simplified_post_process(answer: str, query_type: str, extracted_number: Optional[str], extracted_field_name: Optional[str]) -> str:
    """Simplified post-processing for cases where regular post-processing fails"""
    try:
        # Check if we have a definition for this field in our database
        if extracted_number and has_field_definition(extracted_number):
            return format_field_definition(extracted_number, USE_LINE_BREAKS)
        
        # Check if we can find the field by name
        if extracted_field_name:
            field_id = get_field_id_by_name(extracted_field_name)
            if field_id and has_field_definition(field_id):
                return format_field_definition(field_id, USE_LINE_BREAKS)
                
        # Simple cleaning
        answer = answer.replace("•", "-")

        if USE_LINE_BREAKS:
            # If answer already has newlines in the right format, preserve them
            if "\n**" in answer and answer.count("\n") >= 4:
                return answer.strip()
            
            # Try to extract and format with line breaks
            if " - **" in answer:
                # Split by dash-separated fields and join with newlines
                parts = []
                # Extract title first if it exists
                title_match = re.search(r'(\*\*[\d\.]+\s*[^:*]+:\*\*)', answer)
                if title_match:
                    title = title_match.group(1)
                    parts.append(title)
                    answer = answer[len(title):].strip()
                    if answer.startswith(" - "):
                        answer = answer[3:]
                
                # Process the rest
                field_parts = re.split(r'\s*-\s*\*\*', answer)
                for part in field_parts:
                    if part.strip() and not part.strip().startswith("**"):
                        parts.append(f"**{part.strip()}")
                    elif part.strip():
                        parts.append(part.strip())
                
                return "\n".join(parts)
            else:
                # Add basic formatting for numbered definition
                if query_type == "numbered_definition" and extracted_number and extracted_field_name:
                    return f"""**{extracted_number}. {extracted_field_name}:**
**Format:** N/A
**Technical Name:** N/A
**Length:** N/A
**Positions:** N/A
**Definition:** {answer}"""
                else:
                    return answer
        else:
            # Original dash-based formatting
            # Try to extract and format just the basics
            if "**" in answer and ":" in answer:
                # Already has some formatting, just clean it up
                answer = re.sub(r'\n+', ' ', answer)
                answer = re.sub(r'\s+', ' ', answer)
                return answer.strip()
            else:
                # Add basic formatting for numbered definition
                if query_type == "numbered_definition" and extracted_number and extracted_field_name:
                    return f"**{extracted_number}. {extracted_field_name}:** - **Format:** N/A - **Technical Name:** N/A - **Length:** N/A - **Positions:** N/A - **Definition:** {answer}"
                else:
                    return answer
    except Exception as e:
        logger.error(f"Error in simplified post-processing: {e}")
        # Absolute fallback
        if query_type == "numbered_definition" and extracted_number and extracted_field_name:
            if USE_LINE_BREAKS:
                return f"""**{extracted_number}. {extracted_field_name}:**
Information found but formatting failed. Raw response: {answer[:500]}"""
            else:
                return f"**{extracted_number}. {extracted_field_name}:** - Information found but formatting failed. Raw response: {answer[:500]}"
        else:
            return answer[:1000]  # Limit length for safety

def post_process_answer(answer: str, query_type: str, extracted_number: Optional[str], extracted_field_name: Optional[str]) -> str:
    """Post-process the answer to ensure proper formatting and field values"""
    # Check if we have a definition for this field in our database
    if extracted_number and has_field_definition(extracted_number):
        return format_field_definition(extracted_number, USE_LINE_BREAKS)
    
    # Check if we can find the field by name
    if extracted_field_name:
        field_id = get_field_id_by_name(extracted_field_name)
        if field_id and has_field_definition(field_id):
            return format_field_definition(field_id, USE_LINE_BREAKS)
    
    # No match in our field definitions, process the model's answer
    if USE_LINE_BREAKS:
        # First check if the answer already has line breaks in the right format
        if "\n**" in answer and answer.count("\n") >= 4:
            # Already properly formatted with line breaks
            return answer.strip()
        
        # Replace dash formatting with line breaks
        # First, clean any existing formatting
        answer = re.sub(r'\n+', ' ', answer)
        
        # Convert bullet points to dash format (safer regex)
        answer = answer.replace("• ", "- ")
        answer = answer.replace("•", "-")
        
        # Extract the title separately
        title_pattern = r'(\*\*[\d\.]+\s*[^:*]+:\*\*)'
        title_match = re.search(title_pattern, answer)
        
        if title_match:
            title = title_match.group(1)
            rest_of_answer = answer[len(title_match.group(0)):].strip()
            
            # Replace " - **" with newline and "**"
            rest_of_answer = re.sub(r'\s*-\s*\*\*', '\n**', rest_of_answer)
            
            # Combine title with reformatted content
            answer = f"{title}\n{rest_of_answer}"
        else:
            # If no title match, just replace dashes with newlines
            answer = re.sub(r'\s*-\s*\*\*', '\n**', answer)
        
        # Replace any remaining empty fields
        answer = re.sub(r'(\*\*[^:*]+:\*\*)\s*(?=\n|\*\*|$)', r'\1 N/A', answer)
        
        # Make sure there's a newline after the title if not already present
        if ":**\n" not in answer and ":**" in answer:
            answer = answer.replace(":**", ":**\n", 1)
        
        return answer.strip()
    else:
        # Original dash-based formatting
        # Simplify the formatting by replacing newlines
        answer = re.sub(r'\n+', ' ', answer)
        
        # Convert bullet points to dash format (safer regex)
        answer = answer.replace("• ", "- ")
        answer = answer.replace("•", "-")
        
        # Simple regex to ensure dashes between fields
        answer = re.sub(r'(\*\*[^:*]+:\*\*)\s+(?=\*\*)', r'\1 - ', answer)
        
        # Replace any remaining empty fields
        answer = re.sub(r'(\*\*[^:*]+:\*\*)\s*(?=-|\*\*|$)', r'\1 N/A ', answer)
        
        # Clean up extra spaces
        answer = re.sub(r'\s{2,}', ' ', answer)
        
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
        
        # Check for appropriate formatting elements
        if USE_LINE_BREAKS:
            # Check for newline formatting
            line_count = final_answer.count('\n')
            if line_count >= 4:  # At least 4 newlines separating fields
                validation_score += 0.2
        else:
            # Check for dash-separated format
            dash_count = len(re.findall(r' - \*\*', final_answer))
            if dash_count >= 4:  # At least 4 dashes separating fields
                validation_score += 0.2
        
        # Check for complete definition structure
        definition_elements = ['Format:', 'Technical Name:', 'Length:', 'Position', 'Definition:']
        element_count = sum(1 for element in definition_elements if element in final_answer)
        validation_score += (element_count / len(definition_elements)) * 0.4
        
        # Check for missing values (empty fields)
        missing_pattern = r'\*\*[^:]+:\*\*\s+(N/A|\n)' if USE_LINE_BREAKS else r'\*\*[^:]+:\*\*\s+(-|\*\*)'
        empty_fields = len(re.findall(missing_pattern, final_answer))
        if empty_fields > 0:
            validation_score -= 0.1 * empty_fields
    
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
    if USE_LINE_BREAKS:
        if '\n**' in final_answer:  # Properly formatted with newlines
            validation_score += 0.1
    else:
        if ' - **' in final_answer:  # Properly formatted with dashes
            validation_score += 0.1
    if 'NOTE:' in final_answer:  # Includes important notes
        validation_score += 0.05
    
    # Check for field values
    if USE_LINE_BREAKS:
        field_values = {
            'Format': re.search(r'\*\*Format:\*\*\s*([^\n]+)', final_answer),
            'Technical Name': re.search(r'\*\*Technical Name:\*\*\s*([^\n]+)', final_answer),
            'Length': re.search(r'\*\*Length:\*\*\s*([^\n]+)', final_answer),
            'Positions': re.search(r'\*\*Positions:\*\*\s*([^\n]+)', final_answer)
        }
    else:
        field_values = {
            'Format': re.search(r'\*\*Format:\*\*\s*([^-\n]+)', final_answer),
            'Technical Name': re.search(r'\*\*Technical Name:\*\*\s*([^-\n]+)', final_answer),
            'Length': re.search(r'\*\*Length:\*\*\s*([^-\n]+)', final_answer),
            'Positions': re.search(r'\*\*Positions:\*\*\s*([^-\n]+)', final_answer)
        }
    
    # Check if values are present and not just "N/A"
    for field, match in field_values.items():
        if match and match.group(1).strip() and match.group(1).strip() != "N/A":
            validation_score += 0.05
        elif not match or match.group(1).strip() == "N/A":
            validation_score -= 0.05
    
    # Cap the validation score
    validation_score = min(1.0, max(0.0, validation_score))
    
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
        # Start measuring response time
        import time
        start_time = time.time()
        
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
        
        # Calculate response time
        response_time = round(time.time() - start_time, 2)
        logger.info(f"Query processed in {response_time} seconds")
        
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
            "response_time": response_time,
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