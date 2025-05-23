"""
Aetna Data Science Deep Research Document Processor
Advanced ChromaDB creation system for Universal, External Stop Loss, and Capitation Payment Files
Enhanced with LangChain and intelligent document processing
"""

import os
import sys
import shutil
import json
import time
import logging
import argparse
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

# Document processing imports
import pandas as pd
from dotenv import load_dotenv

# LangChain document loaders
from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    CSVLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.schema import Document

# Vector database
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    # Document settings
    documents_dir: str = "./documents"
    chroma_db_dir: str = "./chroma_db_data"
    collection_name: str = "aetna_docs"
    field_definitions_file: str = "field_definitions.json"
    
    # Processing settings
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 300
    
    # Embedding settings
    embedding_model: str = "text-embedding-005"
    gcp_project_id: str = "aethrag2"
    gcp_location: str = "us-central1"
    
    # Advanced processing
    enable_smart_chunking: bool = True
    enable_field_extraction: bool = True
    enable_table_extraction: bool = True
    recreate_collection: bool = False
    
    # Parallel processing
    batch_size: int = 50
    max_workers: int = 4

# Document type classification patterns
DOCUMENT_TYPE_PATTERNS = {
    "universal_medical_file": [
        r"universal\s+(medical|dental|file|claims?)",
        r"universal\s+specification",
        r"universal\s+format",
        r"UB-04", r"UB04", r"1480",
        r"medical\s+dental\s+file",
        r"universal\s+medical\s+dental"
    ],
    "external_stop_loss": [
        r"external\s+stop\s+loss",
        r"stop\s+loss",
        r"ESL", r"reinsurance",
        r"stop\s*loss\s+report"
    ],
    "capitation_payment": [
        r"capitation\s+payment",
        r"capitation\s+file",
        r"capitation\s+format",
        r"cap\s+payment",
        r"provider\s+payment"
    ],
    "pharmacy_file": [
        r"pharmacy", r"drug", r"rx",
        r"798\s+file", r"pharmacy\s+file",
        r"medication", r"prescription"
    ],
    "eligibility_file": [
        r"eligibility", r"member\s+file",
        r"1000\s+file", r"eligibility\s+format",
        r"member\s+eligibility"
    ],
    "lab_results": [
        r"lab\s+results", r"laboratory",
        r"clinical\s+data", r"lab\s+file",
        r"test\s+results"
    ],
    "data_dictionary": [
        r"data\s+dictionary",
        r"field\s+definitions",
        r"field\s+specifications",
        r"data\s+definitions",
        r"record\s+layout"
    ],
    "file_layout": [
        r"file\s+layout", r"record\s+layout",
        r"file\s+format", r"layout\s+specification",
        r"file\s+structure"
    ],
    "code_tables": [
        r"code\s+table", r"value\s+set",
        r"lookup\s+table", r"reference\s+table",
        r"code\s+values"
    ]
}

# Field definition patterns for extraction
FIELD_PATTERNS = {
    "field_number": [
        r"field\s*#?(\d+)",
        r"(?:pos|position)\s*(\d+)",
        r"(\d+)\s*\.\s*[A-Z]"
    ],
    "field_name": [
        r"field\s+name:\s*([^\n]+)",
        r"name:\s*([^\n]+)",
        r"technical\s+name:\s*([^\n]+)"
    ],
    "field_length": [
        r"length:\s*(\d+)",
        r"size:\s*(\d+)",
        r"(\d+)\s+characters?"
    ],
    "field_format": [
        r"format:\s*([^\n]+)",
        r"type:\s*([^\n]+)",
        r"data\s+type:\s*([^\n]+)"
    ]
}

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': PyPDFLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.xls': UnstructuredExcelLoader,
    '.csv': CSVLoader,
    '.txt': TextLoader,
    '.md': TextLoader
}

class AetnaDeepResearchProcessor:
    """Advanced document processor for Aetna Data Science documentation"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        
        # Set up GCP authentication
        self._setup_gcp_auth()
        
        # Initialize components
        self.embeddings = None
        self.chroma_client = None
        self.collection = None
        self.field_definitions = {}
        
        # Processing statistics
        self.stats = {
            "start_time": datetime.now(),
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "field_definitions_extracted": 0,
            "document_types": defaultdict(int),
            "processing_errors": []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_chroma()
        self._load_field_definitions()
    
    def _setup_gcp_auth(self):
        """Setup GCP authentication"""
        api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.json")
        if os.path.exists(api_key_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key_path
            
            try:
                with open(api_key_path, 'r') as f:
                    key_data = json.load(f)
                    if 'project_id' in key_data and not os.getenv("GOOGLE_CLOUD_PROJECT"):
                        os.environ["GOOGLE_CLOUD_PROJECT"] = key_data['project_id']
                        self.config.gcp_project_id = key_data['project_id']
            except Exception as e:
                logging.warning(f"Could not extract project ID: {e}")
    
    def _setup_logging(self):
        """Configure detailed logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('aetna_document_processing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Initializing Aetna Deep Research Document Processor")
        self.logger.info(f"📁 Documents directory: {self.config.documents_dir}")
        self.logger.info(f"🗄️ ChromaDB directory: {self.config.chroma_db_dir}")
        self.logger.info(f"🤖 Embedding model: {self.config.embedding_model}")
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            self.embeddings = VertexAIEmbeddings(
                model_name=self.config.embedding_model,
                project=self.config.gcp_project_id,
                location=self.config.gcp_location
            )
            self.logger.info(f"✓ Initialized embedding model: {self.config.embedding_model}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config.chroma_db_dir, exist_ok=True)
            
            # Handle recreation if requested
            if self.config.recreate_collection and os.path.exists(self.config.chroma_db_dir):
                self.logger.warning(f"🔄 Recreating ChromaDB at {self.config.chroma_db_dir}")
                try:
                    temp_client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
                    collections = temp_client.list_collections()
                    for collection in collections:
                        if collection.name == self.config.collection_name:
                            temp_client.delete_collection(name=self.config.collection_name)
                            self.logger.info(f"🗑️ Deleted existing collection: {self.config.collection_name}")
                except Exception as e:
                    self.logger.warning(f"Could not delete collection, removing directory: {e}")
                    shutil.rmtree(self.config.chroma_db_dir)
                    os.makedirs(self.config.chroma_db_dir, exist_ok=True)
            
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_db_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.config.collection_name)
                self.logger.info(f"✓ Using existing collection: {self.config.collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.config.collection_name,
                    metadata={
                        "description": "Aetna Data Science Deep Research Collection",
                        "created_at": datetime.now().isoformat(),
                        "version": "2.0"
                    }
                )
                self.logger.info(f"✓ Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise
    
    def _load_field_definitions(self):
        """Load existing field definitions"""
        try:
            if os.path.exists(self.config.field_definitions_file):
                with open(self.config.field_definitions_file, 'r') as f:
                    self.field_definitions = json.load(f)
                self.logger.info(f"✓ Loaded {len(self.field_definitions)} existing field definitions")
            else:
                self.logger.info("📝 No existing field definitions file found")
        except Exception as e:
            self.logger.error(f"❌ Error loading field definitions: {e}")
    
    def classify_document_type(self, content: str, filename: str) -> List[str]:
        """Classify document type based on content and filename"""
        doc_types = []
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower) or re.search(pattern, filename_lower):
                    doc_types.append(doc_type)
                    break
        
        # If no specific type found, classify as general
        if not doc_types:
            doc_types.append("general_documentation")
        
        return doc_types
    
    def extract_field_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract field definitions from document content"""
        field_definitions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for field number patterns
            for pattern in FIELD_PATTERNS["field_number"]:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    field_number = match.group(1)
                    
                    # Extract surrounding context
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 5)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    # Extract additional field information
                    field_info = {
                        "field_id": field_number,
                        "line_number": i + 1,
                        "context": context,
                        "extracted_from": "pattern_matching"
                    }
                    
                    # Try to extract field name
                    for name_pattern in FIELD_PATTERNS["field_name"]:
                        name_match = re.search(name_pattern, context, re.IGNORECASE)
                        if name_match:
                            field_info["field_name"] = name_match.group(1).strip()
                            break
                    
                    # Try to extract field length
                    for length_pattern in FIELD_PATTERNS["field_length"]:
                        length_match = re.search(length_pattern, context, re.IGNORECASE)
                        if length_match:
                            field_info["field_length"] = length_match.group(1)
                            break
                    
                    # Try to extract field format
                    for format_pattern in FIELD_PATTERNS["field_format"]:
                        format_match = re.search(format_pattern, context, re.IGNORECASE)
                        if format_match:
                            field_info["field_format"] = format_match.group(1).strip()
                            break
                    
                    field_definitions.append(field_info)
        
        return field_definitions
    
    def extract_table_data(self, document: Document) -> List[Dict[str, Any]]:
        """Extract structured table data from documents"""
        table_data = []
        content = document.page_content
        
        # Look for table-like structures
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for pipe-separated tables
            if '|' in line and line.count('|') >= 2:
                cells = [cell.strip() for cell in line.split('|')]
                if len(cells) >= 3:  # At least 3 columns
                    table_data.append({
                        "type": "table_row",
                        "line_number": i + 1,
                        "cells": cells,
                        "content": line.strip()
                    })
            
            # Look for tab-separated data
            elif '\t' in line and line.count('\t') >= 2:
                cells = [cell.strip() for cell in line.split('\t')]
                if len(cells) >= 3:
                    table_data.append({
                        "type": "tab_separated",
                        "line_number": i + 1,
                        "cells": cells,
                        "content": line.strip()
                    })
        
        return table_data
    
    def smart_chunk_document(self, document: Document) -> List[Document]:
        """Advanced chunking with content-aware splitting and page tracking"""
        content = document.page_content
        base_metadata = document.metadata
        
        if not self.config.enable_smart_chunking:
            # Use standard recursive splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_documents([document])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": "standard"
                })
            
            return chunks
        
        # Smart chunking based on content structure
        chunks = []
        
        # Split by major sections first
        sections = self._split_by_sections(content)
        total_char_count = 0
        
        for section_idx, section in enumerate(sections):
            # Estimate page number based on character position
            estimated_page = self._estimate_page_from_position(total_char_count, content)
            
            if len(section) <= self.config.max_chunk_size:
                # Section fits in one chunk
                chunk_doc = Document(
                    page_content=section,
                    metadata={
                        **base_metadata,
                        "chunk_type": "section",
                        "section_index": section_idx,
                        "chunk_method": "smart_section",
                        "estimated_page": estimated_page,
                        "section_title": self._extract_section_title(section),
                        "char_start_position": total_char_count,
                        "char_end_position": total_char_count + len(section)
                    }
                )
                chunks.append(chunk_doc)
            else:
                # Section too large, split further
                subsections = self._split_large_section(section)
                subsection_char_count = 0
                
                for subsection_idx, subsection in enumerate(subsections):
                    subsection_page = self._estimate_page_from_position(total_char_count + subsection_char_count, content)
                    
                    chunk_doc = Document(
                        page_content=subsection,
                        metadata={
                            **base_metadata,
                            "chunk_type": "subsection",
                            "section_index": section_idx,
                            "subsection_index": subsection_idx,
                            "chunk_method": "smart_subsection",
                            "estimated_page": subsection_page,
                            "section_title": self._extract_section_title(section),
                            "char_start_position": total_char_count + subsection_char_count,
                            "char_end_position": total_char_count + subsection_char_count + len(subsection)
                        }
                    )
                    chunks.append(chunk_doc)
                    subsection_char_count += len(subsection)
            
            total_char_count += len(section)
        
        # Add final metadata to all chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        return chunks
    
    def _estimate_page_from_position(self, char_position: int, full_content: str) -> int:
        """Estimate page number based on character position in document"""
        # Rough estimate: 2000-3000 characters per page for typical documents
        chars_per_page = 2500
        
        # Look for explicit page markers first
        content_up_to_position = full_content[:char_position] if char_position < len(full_content) else full_content
        
        # Check for common page indicators
        page_markers = re.findall(r'page\s+(\d+)', content_up_to_position.lower())
        if page_markers:
            return int(page_markers[-1])  # Return the last found page number
        
        # Check for form feed characters
        page_breaks = content_up_to_position.count('\f')
        if page_breaks > 0:
            return page_breaks + 1
        
        # Fallback to character-based estimation
        estimated_page = max(1, (char_position // chars_per_page) + 1)
        return estimated_page
    
    def _extract_section_title(self, section: str) -> str:
        """Extract section title from the beginning of a section"""
        lines = section.split('\n')
        
        # Look for title patterns in first few lines
        for line in lines[:3]:
            line = line.strip()
            if line:
                # Check if line looks like a title
                if (len(line) < 100 and 
                    (line.isupper() or 
                     re.match(r'^\d+\.\s+[A-Z]', line) or
                     line.endswith(':'))):
                    return line
        
        # Fallback: return first non-empty line truncated
        for line in lines:
            if line.strip():
                return line.strip()[:50] + "..." if len(line.strip()) > 50 else line.strip()
        
        return "Unknown Section"
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by logical sections"""
        # Look for section headers
        section_patterns = [
            r'\n\s*[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'\n\s*\d+\.\s+[A-Z]',     # Numbered sections
            r'\n\s*[A-Z][a-z]+\s+[A-Z][a-z]+.*:\s*\n',  # Title Case with colon
            r'\n\s*Field\s+\d+',       # Field definitions
        ]
        
        sections = []
        current_section = ""
        lines = content.split('\n')
        
        for line in lines:
            is_section_header = any(re.match(pattern, '\n' + line + '\n') for pattern in section_patterns)
            
            if is_section_header and current_section.strip():
                sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections if sections else [content]
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections using multiple strategies"""
        if len(section) <= self.config.max_chunk_size:
            return [section]
        
        # Try splitting by paragraphs first
        paragraphs = section.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= self.config.max_chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                if len(paragraph) <= self.config.max_chunk_size:
                    current_chunk = paragraph + '\n\n'
                else:
                    # Paragraph too large, split by sentences
                    sentences = re.split(r'[.!?]+\s+', paragraph)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk + sentence) <= self.config.max_chunk_size:
                            current_chunk += sentence + '. '
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + '. '
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [section]
    
    def enhance_metadata(self, document: Document, file_path: str) -> Document:
        """Enhance document metadata with extracted information"""
        content = document.page_content
        filename = os.path.basename(file_path)
        
        # Extract basic file information
        enhanced_metadata = {
            **document.metadata,
            "source_file": filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "processed_at": datetime.now().isoformat(),
            "content_length": len(content),
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:16]
        }
        
        # Classify document type
        doc_types = self.classify_document_type(content, filename)
        enhanced_metadata["document_types"] = doc_types
        enhanced_metadata["primary_doc_type"] = doc_types[0] if doc_types else "unknown"
        
        # Extract field definitions if enabled
        if self.config.enable_field_extraction:
            field_defs = self.extract_field_definitions(content)
            if field_defs:
                enhanced_metadata["contains_field_definitions"] = True
                enhanced_metadata["field_count"] = len(field_defs)
                enhanced_metadata["field_numbers"] = [fd.get("field_id") for fd in field_defs]
            else:
                enhanced_metadata["contains_field_definitions"] = False
        
        # Extract table data if enabled
        if self.config.enable_table_extraction:
            table_data = self.extract_table_data(document)
            if table_data:
                enhanced_metadata["contains_tables"] = True
                enhanced_metadata["table_rows"] = len(table_data)
            else:
                enhanced_metadata["contains_tables"] = False
        
        # Add content analysis
        enhanced_metadata["word_count"] = len(content.split())
        enhanced_metadata["line_count"] = len(content.split('\n'))
        
        # Check for specific Aetna patterns
        aetna_patterns = [
            r"aetna", r"universal\s+file", r"stop\s+loss", 
            r"capitation", r"field\s+\d+", r"ICD", r"CPT"
        ]
        enhanced_metadata["aetna_relevance_score"] = sum(
            1 for pattern in aetna_patterns 
            if re.search(pattern, content, re.IGNORECASE)
        ) / len(aetna_patterns)
        
        return Document(page_content=content, metadata=enhanced_metadata)
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load and process a single document"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in SUPPORTED_EXTENSIONS:
                self.logger.warning(f"⚠️ Unsupported file type: {file_extension} for {file_path}")
                return []
            
            # Get appropriate loader
            loader_class = SUPPORTED_EXTENSIONS[file_extension]
            
            # Special handling for different file types
            if file_extension in ['.xlsx', '.xls']:
                # For Excel files, try to load all sheets
                try:
                    loader = loader_class(file_path)
                    documents = loader.load()
                except Exception:
                    # Fallback to UnstructuredLoader for problematic Excel files
                    loader = UnstructuredLoader(file_path)
                    documents = loader.load()
            elif file_extension == '.csv':
                loader = loader_class(file_path, encoding='utf-8')
                documents = loader.load()
            else:
                loader = loader_class(file_path)
                documents = loader.load()
            
            if not documents:
                self.logger.warning(f"⚠️ No content extracted from {file_path}")
                return []
            
            # Enhance metadata for each document
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = self.enhance_metadata(doc, file_path)
                enhanced_docs.append(enhanced_doc)
            
            self.logger.info(f"✓ Loaded {len(enhanced_docs)} document(s) from {file_path}")
            return enhanced_docs
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {file_path}: {e}")
            self.stats["processing_errors"].append({"file": file_path, "error": str(e)})
            return []
    
    def process_documents(self) -> List[Document]:
        """Process all documents in the documents directory"""
        documents_dir = Path(self.config.documents_dir)
        
        if not documents_dir.exists():
            self.logger.error(f"❌ Documents directory not found: {documents_dir}")
            return []
        
        # Find all supported files
        all_files = []
        for ext in SUPPORTED_EXTENSIONS.keys():
            all_files.extend(documents_dir.glob(f"**/*{ext}"))
        
        self.stats["total_files"] = len(all_files)
        self.logger.info(f"📄 Found {len(all_files)} supported files")
        
        all_documents = []
        
        for file_path in all_files:
            self.logger.info(f"📖 Processing: {file_path.name}")
            
            try:
                # Load document
                docs = self.load_document(str(file_path))
                
                if docs:
                    # Smart chunk each document
                    chunked_docs = []
                    for doc in docs:
                        chunks = self.smart_chunk_document(doc)
                        chunked_docs.extend(chunks)
                    
                    all_documents.extend(chunked_docs)
                    self.stats["processed_files"] += 1
                    self.stats["total_chunks"] += len(chunked_docs)
                    
                    # Update document type statistics
                    for doc in docs:
                        doc_types = doc.metadata.get("document_types", ["unknown"])
                        for doc_type in doc_types:
                            self.stats["document_types"][doc_type] += 1
                    
                    # Extract and store field definitions
                    if self.config.enable_field_extraction:
                        for doc in docs:
                            field_defs = self.extract_field_definitions(doc.page_content)
                            for field_def in field_defs:
                                field_id = field_def.get("field_id")
                                if field_id and field_id not in self.field_definitions:
                                    self.field_definitions[field_id] = field_def
                                    self.stats["field_definitions_extracted"] += 1
                
                else:
                    self.stats["failed_files"] += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing {file_path}: {e}")
                self.stats["failed_files"] += 1
                self.stats["processing_errors"].append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        self.logger.info(f"✓ Processing complete: {self.stats['processed_files']}/{self.stats['total_files']} files, {self.stats['total_chunks']} chunks")
        return all_documents
    
    def create_embeddings_batch(self, documents: List[Document]) -> List[List[float]]:
        """Create embeddings for a batch of documents"""
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for ChromaDB compatibility"""
        sanitized = {}
        
        for key, value in metadata.items():
            # Convert lists to JSON strings
            if isinstance(value, list):
                sanitized[key] = json.dumps(value)
            # Convert datetime objects to ISO strings
            elif hasattr(value, 'isoformat'):
                sanitized[key] = value.isoformat()
            # Convert complex objects to strings
            elif not isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def index_documents(self, documents: List[Document]):
        """Index documents in ChromaDB with batch processing"""
        if not documents:
            self.logger.warning("⚠️ No documents to index")
            return
        
        self.logger.info(f"🔄 Indexing {len(documents)} documents...")
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Create embeddings for batch
                embeddings = self.create_embeddings_batch(batch)
                
                # Prepare data for ChromaDB
                ids = []
                metadatas = []
                documents_content = []
                
                for j, doc in enumerate(batch):
                    # Create unique ID
                    doc_id = f"doc_{i + j}_{doc.metadata.get('content_hash', str(hash(doc.page_content))[:8])}"
                    ids.append(doc_id)
                    
                    # Sanitize metadata
                    sanitized_metadata = self.sanitize_metadata(doc.metadata)
                    metadatas.append(sanitized_metadata)
                    
                    # Document content
                    documents_content.append(doc.page_content)
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents_content,
                    metadatas=metadatas,
                    ids=ids
                )
                
                self.logger.info(f"✓ Indexed batch {i//batch_size + 1}/{(len(documents) - 1)//batch_size + 1}")
                
            except Exception as e:
                self.logger.error(f"❌ Error indexing batch {i//batch_size + 1}: {e}")
                raise
    
    def save_field_definitions(self):
        """Save extracted field definitions to file"""
        if self.field_definitions:
            try:
                with open(self.config.field_definitions_file, 'w') as f:
                    json.dump(self.field_definitions, f, indent=2)
                self.logger.info(f"✓ Saved {len(self.field_definitions)} field definitions to {self.config.field_definitions_file}")
            except Exception as e:
                self.logger.error(f"❌ Error saving field definitions: {e}")
    
    def print_statistics(self):
        """Print processing statistics"""
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]
        
        print("\n" + "="*60)
        print("📊 AETNA DEEP RESEARCH PROCESSING STATISTICS")
        print("="*60)
        print(f"⏱️  Processing time: {duration}")
        print(f"📁 Files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
        print(f"❌ Failed files: {self.stats['failed_files']}")
        print(f"📄 Total chunks created: {self.stats['total_chunks']}")
        print(f"🔧 Field definitions extracted: {self.stats['field_definitions_extracted']}")
        
        print(f"\n📋 Document Types:")
        for doc_type, count in sorted(self.stats['document_types'].items()):
            print(f"   {doc_type}: {count}")
        
        if self.stats['processing_errors']:
            print(f"\n❌ Processing Errors ({len(self.stats['processing_errors'])}):")
            for error in self.stats['processing_errors'][:5]:  # Show first 5
                print(f"   {error['file']}: {error['error']}")
            if len(self.stats['processing_errors']) > 5:
                print(f"   ... and {len(self.stats['processing_errors']) - 5} more")
        
        print("="*60)
    
    def run_full_pipeline(self):
        """Run the complete document processing pipeline"""
        self.logger.info("🚀 Starting Aetna Deep Research Document Processing Pipeline")
        
        try:
            # Process all documents
            documents = self.process_documents()
            
            if not documents:
                self.logger.error("❌ No documents were processed successfully")
                return False
            
            # Index documents in ChromaDB
            self.index_documents(documents)
            
            # Save field definitions
            self.save_field_definitions()
            
            # Print statistics
            self.print_statistics()
            
            self.logger.info("✅ Document processing pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return False

def main():
    """Main function to run the document processor"""
    parser = argparse.ArgumentParser(description="Aetna Deep Research Document Processor")
    
    parser.add_argument("--documents-dir", default="./documents", 
                       help="Directory containing documents to process")
    parser.add_argument("--chroma-dir", default="./chroma_db_data",
                       help="ChromaDB storage directory")
    parser.add_argument("--collection-name", default="aetna_docs",
                       help="ChromaDB collection name")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate the collection (delete existing)")
    parser.add_argument("--chunk-size", type=int, default=1500,
                       help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Chunk overlap for text splitting")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    parser.add_argument("--disable-smart-chunking", action="store_true",
                       help="Disable smart chunking")
    parser.add_argument("--disable-field-extraction", action="store_true",
                       help="Disable field definition extraction")
    parser.add_argument("--disable-table-extraction", action="store_true",
                       help="Disable table data extraction")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProcessingConfig(
        documents_dir=args.documents_dir,
        chroma_db_dir=args.chroma_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
        enable_smart_chunking=not args.disable_smart_chunking,
        enable_field_extraction=not args.disable_field_extraction,
        enable_table_extraction=not args.disable_table_extraction
    )
    
    # Run processor
    processor = AetnaDeepResearchProcessor(config)
    success = processor.run_full_pipeline()
    
    if success:
        print("\n🎉 Document processing completed successfully!")
        print(f"📊 ChromaDB collection '{config.collection_name}' is ready for queries")
    else:
        print("\n💥 Document processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()