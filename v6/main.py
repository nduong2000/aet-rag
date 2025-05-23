"""
Aetna Data Science Deep Research RAG System
Expert Knowledge Engine for Universal, External Stop Loss, and Capitation Payment Files
Built with LangChain and LangGraph for Enhanced Accuracy
"""

import os
import sys
import logging
import json
import time
import traceback
from typing import TypedDict, List, Dict, Optional, Literal, Union, Any, Annotated
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph not available. Using fallback implementation.")
    LANGGRAPH_AVAILABLE = False

# GCP Authentication Setup
API_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.json")
if os.path.exists(API_KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_PATH
    print(f"✓ Using Google credentials from: {API_KEY_PATH}")
    
    try:
        with open(API_KEY_PATH, 'r') as key_file:
            key_data = json.load(key_file)
            if 'project_id' in key_data and not os.getenv("GOOGLE_CLOUD_PROJECT"):
                os.environ["GOOGLE_CLOUD_PROJECT"] = key_data['project_id']
                print(f"✓ Using project ID from key file: {key_data['project_id']}")
    except Exception as e:
        print(f"⚠ Warning: Could not extract project ID from key file: {e}")
else:
    print(f"⚠ Warning: API key file not found at {API_KEY_PATH}")

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # ChromaDB settings
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db_data")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aetna_docs")
    
    # Model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
    DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-pro-preview-05-06")
    AVAILABLE_MODELS = {
        "gemini-2.5-pro-preview-05-06": {"temperature": 0.1, "top_p": 0.8},
        "gemini-2.5-flash-preview-04-17": {"temperature": 0.15, "top_p": 0.9}
    }
    
    # Research settings
    MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "50"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # GCP settings
    GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aethrag2")
    GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    
    # Field definitions
    FIELD_DEFINITIONS_FILE = os.getenv("FIELD_DEFINITIONS_FILE", "field_definitions.json")

# Logging setup
logging.basicConfig(
    level=logging.INFO,  # Changed back from DEBUG to INFO to reduce log verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aetna_rag_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, template_folder='templates')

# State definition for LangGraph
class ResearchState(TypedDict):
    """State for the Deep Research RAG workflow"""
    # Input
    query: str
    model_name: str
    
    # Query Analysis
    query_intent: Literal[
        "field_definition",
        "technical_specification", 
        "data_dictionary",
        "universal_file",
        "external_stop_loss",
        "capitation_payment",
        "file_layout",
        "code_values",
        "eligibility",
        "pharmacy",
        "lab_results",
        "comparative_analysis",
        "general_inquiry"
    ]
    extracted_entities: List[str]
    field_references: List[str]
    technical_terms: List[str]
    
    # Research Strategy
    research_strategies: List[str]
    search_queries: List[str]
    retrieval_methods: List[str]
    
    # Retrieval Results
    retrieved_documents: List[Dict[str, Any]]
    document_scores: List[float]
    filtered_documents: List[Dict[str, Any]]
    
    # Context Building
    primary_context: str
    supporting_context: str
    field_definitions_context: str
    
    # Analysis & Generation
    analysis_notes: str
    confidence_assessment: float
    answer: str
    citations: List[Dict[str, Any]]
    
    # Metadata
    processing_time: float
    tokens_used: int
    research_depth: Literal["surface", "standard", "deep", "comprehensive"]
    error_message: Optional[str]

class AetnaDataScienceRAGSystem:
    """Deep Research RAG System for Aetna Data Science Documentation"""
    
    def __init__(self):
        self.config = Config()
        self.chroma_client = None
        self.collection = None
        self.embeddings = None
        self.field_definitions = {}
        self.workflow = None
        
        # Initialize system
        self._setup_logging()
        self._load_field_definitions()
        self._initialize_chroma()
        self._initialize_embeddings()
        if LANGGRAPH_AVAILABLE:
            self._create_workflow()
    
    def _setup_logging(self):
        """Configure detailed logging"""
        logger.info(f"🚀 Initializing Aetna Data Science RAG System")
        logger.info(f"📊 GCP Project: {self.config.GCP_PROJECT_ID}")
        logger.info(f"🗄️ ChromaDB: {self.config.CHROMA_DB_DIR}")
        logger.info(f"🤖 Default Model: {self.config.DEFAULT_CHAT_MODEL}")
    
    def _load_field_definitions(self):
        """Load field definitions from JSON file"""
        try:
            if os.path.exists(self.config.FIELD_DEFINITIONS_FILE):
                with open(self.config.FIELD_DEFINITIONS_FILE, 'r') as f:
                    self.field_definitions = json.load(f)
                logger.info(f"✓ Loaded {len(self.field_definitions)} field definitions")
            else:
                logger.warning(f"⚠ Field definitions file not found: {self.config.FIELD_DEFINITIONS_FILE}")
        except Exception as e:
            logger.error(f"❌ Error loading field definitions: {e}")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB connection"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.config.COLLECTION_NAME
                )
                logger.info(f"✓ Connected to existing collection: {self.config.COLLECTION_NAME}")
            except Exception:
                logger.error(f"❌ Collection '{self.config.COLLECTION_NAME}' not found. Please run create_chroma_db.py first.")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embeddings = VertexAIEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                project=self.config.GCP_PROJECT_ID,
                location=self.config.GCP_LOCATION
            )
            logger.info(f"✓ Initialized embeddings: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
    
    def _create_workflow(self):
        """Create the LangGraph workflow for deep research"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("filter_and_rank", self._filter_and_rank)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("validate_response", self._validate_response)
        
        # Define flow
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "plan_research")
        workflow.add_edge("plan_research", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "filter_and_rank")
        workflow.add_edge("filter_and_rank", "build_context")
        workflow.add_edge("build_context", "generate_answer")
        workflow.add_edge("generate_answer", "validate_response")
        workflow.add_edge("validate_response", END)
        
        # Compile workflow
        self.workflow = workflow.compile()
        logger.info("✓ Created LangGraph research workflow")
    
    def _analyze_query(self, state: ResearchState) -> ResearchState:
        """Deep analysis of user query to understand intent and extract entities"""
        start_time = time.time()
        
        try:
            # Create analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert Aetna Data Scientist specializing in healthcare data systems. 
                Analyze the user query to understand their intent and extract relevant entities.
                
                Focus on:
                - Field definitions and technical specifications
                - File formats (Universal, External Stop Loss, Capitation)
                - Data dictionary inquiries
                - Code values and lookup tables
                - Technical field names and positions
                
                Return a JSON response with:
                - query_intent: one of the predefined categories
                - extracted_entities: list of key entities/terms
                - field_references: any field names or numbers mentioned
                - technical_terms: healthcare/data science terms
                """),
                ("human", "Query: {query}")
            ])
            
            # Get LLM
            llm = self._get_chat_model(state.get("model_name", self.config.DEFAULT_CHAT_MODEL))
            
            # Analyze query
            chain = analysis_prompt | llm | StrOutputParser()
            response = chain.invoke({"query": state["query"]})
            
            # Parse response (simplified for now)
            state["query_intent"] = self._determine_intent(state["query"])
            state["extracted_entities"] = self._extract_entities(state["query"])
            state["field_references"] = self._extract_field_references(state["query"])
            state["technical_terms"] = self._extract_technical_terms(state["query"])
            
            logger.info(f"✓ Query analyzed - Intent: {state['query_intent']}")
            
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            state["error_message"] = str(e)
            
        state["processing_time"] = time.time() - start_time
        return state
    
    def _plan_research(self, state: ResearchState) -> ResearchState:
        """Plan research strategy based on query analysis"""
        try:
            intent = state["query_intent"]
            
            # Define research strategies based on intent
            strategy_map = {
                "field_definition": ["exact_field_lookup", "semantic_search", "field_definition_context"],
                "technical_specification": ["technical_search", "specification_lookup", "cross_reference"],
                "universal_file": ["universal_file_search", "layout_search", "format_specification"],
                "external_stop_loss": ["stop_loss_search", "reinsurance_context", "report_layout"],
                "capitation_payment": ["capitation_search", "payment_file_context", "provider_data"],
                "file_layout": ["layout_search", "field_position_lookup", "format_specification"],
                "code_values": ["code_table_search", "value_lookup", "reference_tables"],
                "eligibility": ["eligibility_search", "member_data", "coverage_context"],
                "pharmacy": ["pharmacy_search", "drug_data", "rx_file_context"],
                "lab_results": ["lab_search", "clinical_data", "test_results"],
                "comparative_analysis": ["multi_source_search", "comparative_context", "analysis_focus"],
                "general_inquiry": ["broad_search", "contextual_lookup", "general_knowledge"]
            }
            
            state["research_strategies"] = strategy_map.get(intent, ["broad_search", "contextual_lookup"])
            state["search_queries"] = self._generate_search_queries(state)
            state["retrieval_methods"] = ["semantic", "keyword", "hybrid"]
            state["research_depth"] = "deep"  # Default to deep research
            
            logger.info(f"✓ Research planned - Strategies: {len(state['research_strategies'])}")
            
        except Exception as e:
            logger.error(f"❌ Research planning failed: {e}")
            state["error_message"] = str(e)
            
        return state
    
    def _retrieve_documents(self, state: ResearchState) -> ResearchState:
        """Retrieve relevant documents using multiple strategies"""
        try:
            all_documents = []
            all_scores = []
            
            # Semantic retrieval
            if "semantic" in state["retrieval_methods"]:
                semantic_docs, semantic_scores = self._semantic_retrieval(state)
                all_documents.extend(semantic_docs)
                all_scores.extend(semantic_scores)
            
            # Keyword-based retrieval
            if "keyword" in state["retrieval_methods"]:
                keyword_docs, keyword_scores = self._keyword_retrieval(state)
                all_documents.extend(keyword_docs)
                all_scores.extend(keyword_scores)
            
            # Field-specific retrieval
            if state["field_references"]:
                field_docs, field_scores = self._field_specific_retrieval(state)
                all_documents.extend(field_docs)
                all_scores.extend(field_scores)
            
            state["retrieved_documents"] = all_documents
            state["document_scores"] = all_scores
            
            logger.info(f"✓ Retrieved {len(all_documents)} documents")
            
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            state["error_message"] = str(e)
            state["retrieved_documents"] = []
            state["document_scores"] = []
            
        return state
    
    def _filter_and_rank(self, state: ResearchState) -> ResearchState:
        """Filter and rank retrieved documents"""
        try:
            documents = state["retrieved_documents"]
            scores = state["document_scores"]
            
            if not documents:
                state["filtered_documents"] = []
                return state
            
            # Combine documents with scores
            doc_score_pairs = list(zip(documents, scores))
            
            # Log score distribution for debugging
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                logger.info(f"📊 Score distribution: min={min_score:.3f}, max={max_score:.3f}, avg={avg_score:.3f}")
            
            # Use a more lenient threshold for filtering
            # Allow documents with reasonable similarity scores
            adaptive_threshold = min(0.3, self.config.SIMILARITY_THRESHOLD)  # Cap at 0.3 for more lenient filtering
            
            filtered_pairs = [
                (doc, score) for doc, score in doc_score_pairs
                if score >= adaptive_threshold
            ]
            
            # If still too few results, take top documents regardless of threshold
            if len(filtered_pairs) < 5 and len(doc_score_pairs) > 0:
                logger.info(f"⚠️ Only {len(filtered_pairs)} documents passed threshold {adaptive_threshold:.3f}, taking top 10")
                # Sort all documents by score and take top 10
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                filtered_pairs = doc_score_pairs[:10]
            
            # Sort by score (descending)
            filtered_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to max documents
            filtered_pairs = filtered_pairs[:self.config.MAX_RETRIEVAL_DOCS]
            
            # Extract documents
            state["filtered_documents"] = [doc for doc, _ in filtered_pairs]
            
            # Log final filtering results
            if filtered_pairs:
                top_scores = [score for _, score in filtered_pairs[:5]]
                logger.info(f"✓ Filtered to {len(state['filtered_documents'])} relevant documents (top scores: {top_scores})")
            else:
                logger.warning(f"⚠️ No documents passed filtering - threshold may be too high")
            
        except Exception as e:
            logger.error(f"❌ Document filtering failed: {e}")
            state["error_message"] = str(e)
            state["filtered_documents"] = []
            
        return state
    
    def _build_context(self, state: ResearchState) -> ResearchState:
        """Build comprehensive context from filtered documents"""
        try:
            documents = state["filtered_documents"]
            
            if not documents:
                state["primary_context"] = ""
                state["supporting_context"] = ""
                state["field_definitions_context"] = ""
                return state
            
            # Categorize documents with source tracking
            primary_docs = []
            supporting_docs = []
            field_def_docs = []
            
            for doc in documents:
                doc_type = doc.get("metadata", {}).get("document_type", "general")
                content = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
                source_name = metadata.get("source_file", "Unknown Document")
                page_info = f", Page {metadata.get('page')}" if metadata.get('page') else ""
                
                # Add source attribution to content
                attributed_content = f"[Source: {source_name}{page_info}]\n{content}"
                
                if any(term in content.lower() for term in state["extracted_entities"]):
                    primary_docs.append(attributed_content)
                elif "field" in content.lower() or "definition" in content.lower():
                    field_def_docs.append(attributed_content)
                else:
                    supporting_docs.append(attributed_content)
            
            # Build context sections with clear attribution
            if primary_docs:
                state["primary_context"] = "\n\n---\n\n".join(primary_docs[:10])  # Top 10 primary docs
            else:
                state["primary_context"] = ""
                
            if supporting_docs:
                state["supporting_context"] = "\n\n---\n\n".join(supporting_docs[:5])  # Top 5 supporting docs
            else:
                state["supporting_context"] = ""
                
            if field_def_docs:
                state["field_definitions_context"] = "\n\n---\n\n".join(field_def_docs[:5])  # Top 5 field docs
            else:
                state["field_definitions_context"] = ""
            
            # Add field definitions from loaded data
            if state["field_references"]:
                field_context = []
                for field_ref in state["field_references"]:
                    if field_ref in self.field_definitions:
                        field_info = self.field_definitions[field_ref]
                        field_context.append(f"[Source: Field Definitions Database]\nField {field_ref}: {field_info}")
                
                if field_context:
                    if state["field_definitions_context"]:
                        state["field_definitions_context"] += "\n\n---\n\n" + "\n\n---\n\n".join(field_context)
                    else:
                        state["field_definitions_context"] = "\n\n---\n\n".join(field_context)
            
            logger.info(f"✓ Built context - Primary: {len(primary_docs)} docs, Supporting: {len(supporting_docs)} docs, Field defs: {len(field_def_docs)} docs")
            
        except Exception as e:
            logger.error(f"❌ Context building failed: {e}")
            state["error_message"] = str(e)
            state["primary_context"] = ""
            state["supporting_context"] = ""
            state["field_definitions_context"] = ""
            
        return state
    
    def _generate_answer(self, state: ResearchState) -> ResearchState:
        """Generate comprehensive answer using Aetna Data Scientist persona"""
        try:
            # Build the expert prompt
            prompt = self._build_expert_prompt(state)
            
            # Get LLM
            llm = self._get_chat_model(state.get("model_name", self.config.DEFAULT_CHAT_MODEL))
            
            # Generate response
            response = llm.invoke(prompt)
            answer = response.content
            
            # Extract citations for structured display (frontend will handle these)
            state["citations"] = self._extract_citations(state)
            
            # Store clean answer without text citations (frontend handles citation display)
            state["answer"] = answer
            
            # Calculate confidence based on context quality and response coherence
            state["confidence_assessment"] = self._assess_confidence(state)
            
            logger.info(f"✓ Generated answer with {len(state['citations'])} citations - Confidence: {state['confidence_assessment']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            state["error_message"] = str(e)
            state["answer"] = "I apologize, but I encountered an error while generating the response. Please try again."
            state["confidence_assessment"] = 0.0
            state["citations"] = []
            
        return state
    
    def _validate_response(self, state: ResearchState) -> ResearchState:
        """Validate and enhance the response"""
        try:
            # Check for minimum confidence threshold
            if state["confidence_assessment"] < 0.5:
                state["answer"] += "\n\n⚠️ Note: This response has lower confidence. Please verify with official documentation."
            
            # Add metadata
            state["tokens_used"] = len(state["answer"].split())  # Simplified token count
            
            logger.info(f"✓ Response validated - Final confidence: {state['confidence_assessment']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Response validation failed: {e}")
            
        return state
    
    def _get_chat_model(self, model_name: str = None):
        """Get configured chat model"""
        model_name = model_name or self.config.DEFAULT_CHAT_MODEL
        model_config = self.config.AVAILABLE_MODELS.get(model_name, {"temperature": 0.1})
        
        return ChatVertexAI(
            model_name=model_name,
            project=self.config.GCP_PROJECT_ID,
            location=self.config.GCP_LOCATION,
            **model_config
        )
    
    def _build_expert_prompt(self, state: ResearchState) -> List:
        """Build expert prompt for Aetna Data Scientist persona"""
        system_message = """You are a senior Aetna Data Scientist with deep expertise in healthcare data systems, 
        particularly Universal Medical/Dental Files, External Stop Loss reports, and Capitation Payment files.

        Your role:
        - Provide accurate, detailed responses about Aetna's data specifications
        - Reference specific field definitions, layouts, and technical specifications
        - Explain complex healthcare data concepts clearly
        - ALWAYS cite relevant documentation with specific details when available
        - Maintain professional healthcare data analysis standards

        Citation Guidelines:
        - When referencing specific information, mention the source document name
        - Include page numbers when available (e.g., "According to the Universal File Layout, page 15...")
        - Reference specific field numbers and positions when discussing fields
        - Use phrases like "As documented in [document name]" or "Per the [file type] specification"
        - Be specific about which documents contain which information

        Response Structure:
        - Start with a direct answer to the question
        - Provide detailed technical information with source references
        - Include specific field numbers, positions, and technical names when available
        - Explain both the technical and business context
        - Highlight any important caveats or considerations
        - End with additional relevant information if applicable
        """
        
        # Build context summary for the prompt
        context_summary = ""
        
        if state.get('primary_context'):
            context_summary += f"PRIMARY DOCUMENTATION:\n{state['primary_context']}\n\n"
        
        if state.get('field_definitions_context'):
            context_summary += f"FIELD DEFINITIONS:\n{state['field_definitions_context']}\n\n"
        
        if state.get('supporting_context'):
            context_summary += f"SUPPORTING INFORMATION:\n{state['supporting_context']}\n\n"
        
        human_message = f"""
        Query: {state['query']}
        
        Available Documentation Context:
        {context_summary if context_summary else 'No specific documentation context available'}
        
        Instructions:
        - Answer the question comprehensively using the provided documentation
        - Cite specific document names and page numbers when referencing information
        - Include field numbers and technical specifications when relevant
        - If referencing field definitions, mention the field number and name
        - Structure your response for healthcare data analysts and professionals
        - If the documentation doesn't fully answer the question, state what additional information would be needed
        
        Please provide your expert response with proper source citations throughout.
        """
        
        return [SystemMessage(content=system_message), HumanMessage(content=human_message)]
    
    # Helper methods for analysis and retrieval
    def _determine_intent(self, query: str) -> str:
        """Determine query intent from text analysis"""
        query_lower = query.lower()
        
        intent_keywords = {
            "field_definition": ["field", "definition", "what is", "define", "meaning"],
            "universal_file": ["universal", "medical file", "dental file", "1480", "dental claim", "medical claim"],
            "external_stop_loss": ["stop loss", "external", "esl", "reinsurance"],
            "capitation_payment": ["capitation", "payment", "provider payment"],
            "file_layout": ["layout", "format", "structure", "positions"],
            "code_values": ["code", "values", "table", "lookup"],
            "eligibility": ["eligibility", "member", "coverage"],
            "pharmacy": ["pharmacy", "drug", "rx", "798"],
            "lab_results": ["lab", "test", "clinical", "results"],
            "data_dictionary": ["identify", "how to", "dental claim", "medical claim", "claim identification"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "general_inquiry"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query"""
        # Simplified entity extraction
        entities = []
        
        # Common healthcare data terms
        healthcare_terms = [
            "diagnosis", "procedure", "claim", "member", "provider", "coverage",
            "icd", "cpt", "npi", "billing", "payment", "eligibility", "benefits"
        ]
        
        query_lower = query.lower()
        for term in healthcare_terms:
            if term in query_lower:
                entities.append(term)
        
        return entities
    
    def _extract_field_references(self, query: str) -> List[str]:
        """Extract field references from query"""
        import re
        
        field_refs = []
        
        # Look for field numbers
        field_numbers = re.findall(r'field\s*(\d+)', query.lower())
        field_refs.extend(field_numbers)
        
        # Look for specific field patterns
        field_patterns = re.findall(r'\b([A-Z_]+_[A-Z_]+)\b', query)
        field_refs.extend(field_patterns)
        
        return field_refs
    
    def _extract_technical_terms(self, query: str) -> List[str]:
        """Extract technical terms from query"""
        technical_terms = []
        
        # Common technical terms in healthcare data
        tech_keywords = [
            "varchar", "numeric", "date", "timestamp", "position", "length",
            "format", "specification", "layout", "record", "file"
        ]
        
        query_lower = query.lower()
        for term in tech_keywords:
            if term in query_lower:
                technical_terms.append(term)
        
        return technical_terms
    
    def _generate_search_queries(self, state: ResearchState) -> List[str]:
        """Generate search queries based on analysis"""
        base_query = state["query"]
        entities = state["extracted_entities"]
        field_refs = state["field_references"]
        
        queries = [base_query]
        
        # Add entity-focused queries
        for entity in entities:
            queries.append(f"{entity} definition specification")
            queries.append(f"{entity} field layout")
        
        # Add field-specific queries
        for field_ref in field_refs:
            queries.append(f"field {field_ref} definition")
            queries.append(f"{field_ref} specification")
        
        return queries[:10]  # Limit to 10 queries
    
    def _semantic_retrieval(self, state: ResearchState) -> tuple:
        """Perform semantic retrieval using embeddings"""
        if not self.collection:
            return [], []
        
        try:
            query = state["query"]
            
            # Generate embedding for the query using our VertexAI model
            query_embedding = self.embeddings.embed_query(query)
            
            # Use the embedding directly instead of query_texts
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(30, self.config.MAX_RETRIEVAL_DOCS),
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            scores = []
            
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    score = 1.0 - distance  # Convert distance to similarity
                    
                    documents.append({
                        "page_content": doc,
                        "metadata": metadata
                    })
                    scores.append(score)
            
            return documents, scores
            
        except Exception as e:
            logger.error(f"❌ Semantic retrieval failed: {e}")
            return [], []
    
    def _keyword_retrieval(self, state: ResearchState) -> tuple:
        """Perform keyword-based retrieval"""
        # Simplified keyword retrieval
        return self._semantic_retrieval(state)  # Fallback to semantic for now
    
    def _field_specific_retrieval(self, state: ResearchState) -> tuple:
        """Retrieve documents specific to mentioned fields"""
        if not self.collection or not state["field_references"]:
            return [], []
        
        try:
            all_docs = []
            all_scores = []
            
            for field_ref in state["field_references"]:
                field_query = f"field {field_ref}"
                
                # Generate embedding for the field query using our VertexAI model
                field_query_embedding = self.embeddings.embed_query(field_query)
                
                results = self.collection.query(
                    query_embeddings=[field_query_embedding],
                    n_results=10,
                    include=["documents", "metadatas", "distances"]
                )
                
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        distance = results["distances"][0][i] if results["distances"] else 1.0
                        score = 1.0 - distance
                        
                        all_docs.append({
                            "page_content": doc,
                            "metadata": metadata
                        })
                        all_scores.append(score)
            
            return all_docs, all_scores
            
        except Exception as e:
            logger.error(f"❌ Field-specific retrieval failed: {e}")
            return [], []
    
    def _assess_confidence(self, state: ResearchState) -> float:
        """Assess confidence in the generated response"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available context
        if state.get("primary_context"):
            confidence += 0.2
        
        if state.get("field_definitions_context"):
            confidence += 0.2
        
        if state.get("supporting_context"):
            confidence += 0.1
        
        # Adjust based on query intent match
        if state["query_intent"] != "general_inquiry":
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize metadata arrays that were stored as JSON strings"""
        deserialized = metadata.copy()
        
        # List of fields that should be arrays but might be stored as JSON strings
        array_fields = ['field_numbers', 'document_types']
        
        for field in array_fields:
            if field in deserialized and isinstance(deserialized[field], str):
                try:
                    # Try to parse JSON string back to array
                    parsed_value = json.loads(deserialized[field])
                    if isinstance(parsed_value, list):
                        deserialized[field] = parsed_value
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, leave as string and log warning
                    logger.warning(f"Failed to deserialize {field}: {deserialized[field]}")
        
        return deserialized
    
    def _extract_citations(self, state: ResearchState) -> List[Dict[str, Any]]:
        """Extract comprehensive citation information from used documents"""
        citations = []
        seen_sources = set()  # Avoid duplicate citations
        
        for i, doc in enumerate(state.get("filtered_documents", [])[:10]):  # Top 10 documents
            metadata = doc.get("metadata", {})
            
            # Deserialize any JSON-stringified arrays
            metadata = self._deserialize_metadata(metadata)
            
            # Extract detailed source information with multiple fallbacks
            source_file = (metadata.get("source_file") or 
                          metadata.get("source") or 
                          metadata.get("source_filename") or 
                          "Unknown Document")
            
            file_path = metadata.get("file_path", "")
            
            # Use multiple page sources with priority
            page_num = (metadata.get("page") or 
                       metadata.get("estimated_page") or 
                       metadata.get("chunk_index", 0) + 1)
            
            doc_type = metadata.get("primary_doc_type", metadata.get("document_type", "Unknown"))
            section_title = metadata.get("section_title", "")
            
            # Create a more specific identifier for this source
            source_key = f"{source_file}_{page_num}_{metadata.get('chunk_index', 0)}"
            
            if source_key not in seen_sources:
                citation = {
                    "index": len(citations) + 1,
                    "source_file": source_file,
                    "document_type": doc_type,
                    "file_path": file_path,
                    "relevance_rank": i + 1,
                    "content_length": metadata.get("content_length", 0),
                    "aetna_relevance_score": metadata.get("aetna_relevance_score", 0.0),
                    "contains_field_definitions": metadata.get("contains_field_definitions", False),
                    "field_numbers": metadata.get("field_numbers", []),
                    "chunk_method": metadata.get("chunk_method", "unknown")
                }
                
                # Build formatted source with available information
                formatted_parts = [source_file]
                
                if page_num and page_num > 0:
                    if metadata.get("estimated_page"):
                        formatted_parts.append(f"~Page {page_num}")  # Estimated page
                    else:
                        formatted_parts.append(f"Page {page_num}")
                
                if section_title and section_title != "Unknown Section":
                    formatted_parts.append(f"Section: {section_title}")
                
                citation["page"] = page_num
                citation["section_title"] = section_title
                citation["formatted_source"] = ", ".join(formatted_parts)
                
                # Add position information if available
                if metadata.get("char_start_position") is not None:
                    citation["char_position"] = metadata["char_start_position"]
                
                citations.append(citation)
                seen_sources.add(source_key)
        
        logger.info(f"📚 Generated {len(citations)} citations")
        return citations
    
    def _format_citations_text(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations as text for inclusion in the response"""
        if not citations:
            return ""
        
        citation_text = "\n\n**Sources:**\n"
        for cite in citations:
            citation_text += f"[{cite['index']}] {cite['formatted_source']}"
            
            # Add document type information
            if cite['document_type'] != "Unknown":
                citation_text += f" ({cite['document_type']})"
            
            # Add field information if available
            if cite.get('contains_field_definitions') and cite.get('field_numbers'):
                field_nums = cite['field_numbers'][:3]  # Show first 3 field numbers
                field_text = ", ".join(str(f) for f in field_nums)
                if len(cite['field_numbers']) > 3:
                    field_text += f" and {len(cite['field_numbers']) - 3} more"
                citation_text += f" - Contains fields: {field_text}"
            
            citation_text += "\n"
        
        return citation_text
    
    def process_query(self, query: str, model_name: str = None) -> Dict[str, Any]:
        """Process a query through the deep research workflow"""
        start_time = time.time()
        
        try:
            if not LANGGRAPH_AVAILABLE or not self.workflow:
                return self._fallback_processing(query, model_name)
            
            # Initial state
            initial_state = {
                "query": query,
                "model_name": model_name or self.config.DEFAULT_CHAT_MODEL,
                "processing_time": 0.0,
                "tokens_used": 0,
                "research_depth": "deep",
                "error_message": None
            }
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            final_state["processing_time"] = total_time
            
            # Format response
            return {
                "answer": final_state.get("answer", "No answer generated"),
                "confidence_score": final_state.get("confidence_assessment", 0.0),
                "query_intent": final_state.get("query_intent", "unknown"),
                "research_strategies": final_state.get("research_strategies", []),
                "citations": final_state.get("citations", []),
                "retrieved_sources_metadata": final_state.get("citations", []),
                "model_used": final_state.get("model_name", self.config.DEFAULT_CHAT_MODEL),
                "response_time": round(total_time, 2),
                "tokens_used": final_state.get("tokens_used", 0),
                "research_depth": final_state.get("research_depth", "deep"),
                "error": final_state.get("error_message")
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            traceback.print_exc()
            
            return {
                "answer": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    def _fallback_processing(self, query: str, model_name: str = None) -> Dict[str, Any]:
        """Fallback processing when LangGraph is not available"""
        start_time = time.time()
        
        try:
            # Simple semantic retrieval
            if self.collection:
                # Generate embedding for the query using our VertexAI model
                query_embedding = self.embeddings.embed_query(query)
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=20,
                    include=["documents", "metadatas", "distances"]
                )
                
                context = ""
                citations = []
                context_docs = []
                
                if results["documents"] and results["documents"][0]:
                    # Build context and extract citation information
                    for i, doc in enumerate(results["documents"][0][:10]):
                        context += f"{doc}\n\n"
                        
                        # Extract metadata for citations
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        
                        # Deserialize any JSON-stringified arrays
                        metadata = self._deserialize_metadata(metadata)
                        
                        distance = results["distances"][0][i] if results["distances"] else 1.0
                        
                        context_docs.append({
                            "page_content": doc,
                            "metadata": metadata
                        })
                
                # Extract citations using the same method as the main workflow
                if context_docs:
                    dummy_state = {"filtered_documents": context_docs}
                    citations = self._extract_citations(dummy_state)
                
                # Generate response with enhanced prompt
                llm = self._get_chat_model(model_name)
                
                prompt = f"""You are an expert Aetna Data Scientist with deep knowledge of healthcare data systems.

Query: {query}

Available Documentation Context:
{context}

Instructions:
- Provide a comprehensive, accurate response using the provided documentation
- When referencing specific information, cite the source documents when possible
- Include field numbers and technical specifications when mentioned in the context
- Explain both technical and business aspects clearly
- If the context doesn't fully answer the question, acknowledge the limitation

Please provide your expert response with appropriate source references."""
                
                response = llm.invoke([HumanMessage(content=prompt)])
                answer = response.content
                
                # Store clean answer without text citations (frontend handles citation display)
                # citation_text = self._format_citations_text(citations)
                # if citation_text:
                #     answer += citation_text
                
                return {
                    "answer": answer,
                    "confidence_score": 0.7,
                    "query_intent": "general",
                    "research_strategies": ["semantic_search"],
                    "citations": citations,
                    "retrieved_sources_metadata": citations,
                    "model_used": model_name or self.config.DEFAULT_CHAT_MODEL,
                    "response_time": round(time.time() - start_time, 2),
                    "tokens_used": len(answer.split()),
                    "research_depth": "standard",
                    "error": None
                }
            else:
                return {
                    "answer": "ChromaDB collection not available. Please ensure the database is properly initialized.",
                    "confidence_score": 0.0,
                    "error": "Database not available"
                }
                
        except Exception as e:
            logger.error(f"❌ Fallback processing failed: {e}")
            return {
                "answer": f"Error in fallback processing: {str(e)}",
                "confidence_score": 0.0,
                "error": str(e)
            }

# Initialize the system
rag_system = AetnaDataScienceRAGSystem()

# Flask routes
@app.route('/')
def chat_interface():
    """Serve the chat interface"""
    return render_template('chat.html')

@app.route('/chat_rag', methods=['POST'])
def chat_rag():
    """Main chat endpoint for the RAG system"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query'].strip()
        model_name = data.get('model', rag_system.config.DEFAULT_CHAT_MODEL)
        
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        # Process the query
        result = rag_system.process_query(query, model_name)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "answer": "I apologize, but I encountered an error. Please try again."
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        status = {
            "status": "healthy",
            "chroma_db": bool(rag_system.collection),
            "embeddings": bool(rag_system.embeddings),
            "workflow": bool(rag_system.workflow) if LANGGRAPH_AVAILABLE else "fallback",
            "field_definitions": len(rag_system.field_definitions),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Aetna Data Science Deep Research RAG System")
    app.run(host='0.0.0.0', port=5000, debug=False)