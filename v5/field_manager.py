#!/usr/bin/env python3
"""
Field Definition Manager - Utility for managing medical field definitions

This script provides functions to:
1. Add new field definitions 
2. Export current definitions to JSON
3. Import definitions from JSON
4. Generate field mapping from names to numbers
5. Parse field definitions from Word document
"""

import json
import os
import sys
import argparse
import re
from docx import Document

# Default field definitions file
DEFAULT_FIELD_FILE = "field_definitions.json"
DEFAULT_DOCX_FILE = "documents/Aetna-universal-medical-dental-data-dictionary.docx"

def load_field_definitions(file_path):
    """Load field definitions from a JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}
    except Exception as e:
        print(f"Error loading field definitions: {e}")
        return {}

def save_field_definitions(definitions, file_path):
    """Save field definitions to a JSON file"""
    try:
        with open(file_path, 'w') as file:
            json.dump(definitions, file, indent=2)
        print(f"Saved {len(definitions)} field definitions to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving field definitions: {e}")
        return False

def add_field(definitions, field_id, title, format_value, tech_name, length, positions, definition):
    """Add a new field definition"""
    definitions[field_id] = {
        "title": title,
        "format": format_value,
        "technical_name": tech_name,
        "length": length,
        "positions": positions,
        "definition": definition
    }
    return definitions

def update_field(definitions, field_id, **kwargs):
    """Update an existing field definition"""
    if field_id not in definitions:
        print(f"Field {field_id} not found.")
        return definitions
    
    for key, value in kwargs.items():
        if value is not None and key in definitions[field_id]:
            definitions[field_id][key] = value
    
    return definitions

def delete_field(definitions, field_id):
    """Delete a field definition"""
    if field_id in definitions:
        del definitions[field_id]
        print(f"Deleted field {field_id}.")
    else:
        print(f"Field {field_id} not found.")
    return definitions

def generate_field_mapping(definitions):
    """Generate a mapping from field names to field IDs"""
    mapping = {}
    for field_id, field_data in definitions.items():
        title = field_data.get("title", "").lower()
        if title:
            mapping[title] = field_id
            
            # Add common variations
            words = title.split()
            if len(words) > 1:
                # Add without "code" if it ends with "code"
                if words[-1] == "code":
                    mapping[" ".join(words[:-1])] = field_id
                
                # Add technical name-like version
                tech_name = title.replace(" ", "_").upper()
                mapping[tech_name] = field_id
    
    return mapping

def parse_word_document(docx_path, start_field=1, end_field=178):
    """
    Parse field definitions from a Word document
    
    Args:
        docx_path: Path to the .docx file
        start_field: First field number to extract
        end_field: Last field number to extract
        
    Returns:
        Dictionary of field definitions
    """
    print(f"Parsing Word document: {docx_path}")
    if not os.path.exists(docx_path):
        print(f"Error: Document not found at {docx_path}")
        return {}
        
    try:
        document = Document(docx_path)
        
        # Dictionary to store field definitions
        field_definitions = {}
        
        # Regex patterns to extract field information
        field_pattern = re.compile(r'^\s*(\d+)\.\s+(.+?)(?:\s*\(|$)')
        format_pattern = re.compile(r'Format:\s*(Character|Numeric|Date|Alpha-Numeric|Text)')
        tech_name_pattern = re.compile(r'Technical Name:\s*([A-Z][A-Z0-9_]+)')
        length_pattern = re.compile(r'Length:\s*(\d+)')
        positions_pattern = re.compile(r'Positions?:\s*(\d+(?:-\d+)?)')
        
        # Current field being processed
        current_field = None
        current_title = None
        current_format = None
        current_tech_name = None
        current_length = None
        current_positions = None
        current_definition = []
        in_definition = False
        
        # Process each paragraph
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            
            if not text:
                continue
                
            # Check if this is a new field
            field_match = field_pattern.match(text)
            if field_match:
                # Save previous field if any
                if current_field and current_title:
                    field_id = str(current_field)
                    if int(field_id) >= start_field and int(field_id) <= end_field:
                        field_definitions[field_id] = {
                            "title": current_title,
                            "format": current_format or "Unknown",
                            "technical_name": current_tech_name or "Unknown",
                            "length": current_length or "Unknown",
                            "positions": current_positions or "Unknown",
                            "definition": "\n".join(current_definition) if current_definition else "No definition available"
                        }
                        print(f"Extracted field {field_id}: {current_title}")
                
                # Start new field
                current_field = int(field_match.group(1))
                current_title = field_match.group(2).strip()
                current_format = None
                current_tech_name = None
                current_length = None
                current_positions = None
                current_definition = []
                in_definition = False
                
            # Look for format
            format_match = format_pattern.search(text)
            if format_match:
                current_format = format_match.group(1)
                
            # Look for technical name
            tech_name_match = tech_name_pattern.search(text)
            if tech_name_match:
                current_tech_name = tech_name_match.group(1)
                
            # Look for length
            length_match = length_pattern.search(text)
            if length_match:
                current_length = length_match.group(1)
                
            # Look for positions
            positions_match = positions_pattern.search(text)
            if positions_match:
                current_positions = positions_match.group(1)
                
            # Check if this is the definition section
            if "Definition:" in text:
                in_definition = True
                definition_text = text.split("Definition:", 1)[1].strip()
                if definition_text:
                    current_definition.append(definition_text)
            elif in_definition and text:
                # Continue adding to definition
                current_definition.append(text)
        
        # Add the last field
        if current_field and current_title:
            field_id = str(current_field)
            if int(field_id) >= start_field and int(field_id) <= end_field:
                field_definitions[field_id] = {
                    "title": current_title,
                    "format": current_format or "Unknown",
                    "technical_name": current_tech_name or "Unknown",
                    "length": current_length or "Unknown",
                    "positions": current_positions or "Unknown",
                    "definition": "\n".join(current_definition) if current_definition else "No definition available"
                }
                print(f"Extracted field {field_id}: {current_title}")
        
        print(f"Extracted {len(field_definitions)} field definitions from document")
        return field_definitions
        
    except Exception as e:
        print(f"Error parsing Word document: {e}")
        import traceback
        traceback.print_exc()
        return {}

def interactive_add():
    """Interactive mode to add a new field definition"""
    field_id = input("Field ID: ")
    title = input("Field Title: ")
    format_value = input("Format (Character/Numeric/Date): ")
    tech_name = input("Technical Name: ")
    length = input("Length: ")
    positions = input("Positions: ")
    print("Definition (enter a blank line to finish):")
    
    definition_lines = []
    while True:
        line = input()
        if not line.strip():
            break
        definition_lines.append(line)
    
    definition = "\n".join(definition_lines)
    
    return field_id, {
        "title": title,
        "format": format_value,
        "technical_name": tech_name,
        "length": length,
        "positions": positions,
        "definition": definition
    }

def main():
    parser = argparse.ArgumentParser(description="Manage medical field definitions")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add field command
    add_parser = subparsers.add_parser("add", help="Add a new field definition")
    add_parser.add_argument("--field-id", required=True, help="Field ID (number)")
    add_parser.add_argument("--title", required=True, help="Field title")
    add_parser.add_argument("--format", required=True, help="Field format (Character/Numeric/Date)")
    add_parser.add_argument("--tech-name", required=True, help="Technical name")
    add_parser.add_argument("--length", required=True, help="Field length")
    add_parser.add_argument("--positions", required=True, help="Field positions")
    add_parser.add_argument("--definition", required=True, help="Field definition")
    add_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions file")
    
    # Interactive add command
    interactive_parser = subparsers.add_parser("interactive", help="Add a field definition interactively")
    interactive_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions file")
    
    # Update field command
    update_parser = subparsers.add_parser("update", help="Update an existing field definition")
    update_parser.add_argument("--field-id", required=True, help="Field ID to update")
    update_parser.add_argument("--title", help="New field title")
    update_parser.add_argument("--format", help="New field format")
    update_parser.add_argument("--tech-name", help="New technical name")
    update_parser.add_argument("--length", help="New field length")
    update_parser.add_argument("--positions", help="New field positions")
    update_parser.add_argument("--definition", help="New field definition")
    update_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions file")
    
    # Delete field command
    delete_parser = subparsers.add_parser("delete", help="Delete a field definition")
    delete_parser.add_argument("--field-id", required=True, help="Field ID to delete")
    delete_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions file")
    
    # Generate mapping command
    mapping_parser = subparsers.add_parser("generate-mapping", help="Generate field name to ID mapping")
    mapping_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions file")
    mapping_parser.add_argument("--output", default="field_name_mapping.py", help="Output Python file")
    
    # List fields command
    list_parser = subparsers.add_parser("list", help="List all field definitions")
    list_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions file")
    list_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    # Parse docx command
    parse_parser = subparsers.add_parser("parse-docx", help="Parse fields from Word document")
    parse_parser.add_argument("--docx", default=DEFAULT_DOCX_FILE, help="Word document to parse")
    parse_parser.add_argument("--start", type=int, default=1, help="Starting field number")
    parse_parser.add_argument("--end", type=int, default=178, help="Ending field number")
    parse_parser.add_argument("--file", default=DEFAULT_FIELD_FILE, help="Field definitions output file")
    parse_parser.add_argument("--merge", action="store_true", help="Merge with existing definitions")
    
    args = parser.parse_args()
    
    if args.command == "add":
        definitions = load_field_definitions(args.file)
        definitions = add_field(
            definitions, 
            args.field_id, 
            args.title, 
            args.format, 
            args.tech_name, 
            args.length, 
            args.positions, 
            args.definition
        )
        save_field_definitions(definitions, args.file)
    
    elif args.command == "interactive":
        definitions = load_field_definitions(args.file)
        field_id, field_data = interactive_add()
        definitions[field_id] = field_data
        save_field_definitions(definitions, args.file)
    
    elif args.command == "update":
        definitions = load_field_definitions(args.file)
        definitions = update_field(
            definitions,
            args.field_id,
            title=args.title,
            format=args.format,
            technical_name=args.tech_name,
            length=args.length,
            positions=args.positions,
            definition=args.definition
        )
        save_field_definitions(definitions, args.file)
    
    elif args.command == "delete":
        definitions = load_field_definitions(args.file)
        definitions = delete_field(definitions, args.field_id)
        save_field_definitions(definitions, args.file)
    
    elif args.command == "generate-mapping":
        definitions = load_field_definitions(args.file)
        mapping = generate_field_mapping(definitions)
        
        # Write to Python file
        with open(args.output, 'w') as f:
            f.write("# Auto-generated field name to ID mapping\n")
            f.write("FIELD_NAME_TO_NUMBER = {\n")
            for name, field_id in sorted(mapping.items()):
                f.write(f'    "{name}": "{field_id}",\n')
            f.write("}\n")
        
        print(f"Generated mapping with {len(mapping)} entries to {args.output}")
    
    elif args.command == "list":
        definitions = load_field_definitions(args.file)
        
        if args.format == "json":
            print(json.dumps(definitions, indent=2))
        else:
            for field_id, field_data in sorted(definitions.items(), key=lambda x: int(x[0])):
                title = field_data.get("title", "")
                format_value = field_data.get("format", "")
                tech_name = field_data.get("technical_name", "")
                print(f"{field_id}: {title} ({format_value}, {tech_name})")
    
    elif args.command == "parse-docx":
        if args.merge:
            existing_definitions = load_field_definitions(args.file)
            print(f"Loaded {len(existing_definitions)} existing field definitions")
            parsed_definitions = parse_word_document(args.docx, args.start, args.end)
            
            # Merge definitions, giving priority to newly parsed ones
            merged_definitions = {**existing_definitions, **parsed_definitions}
            print(f"Merged into {len(merged_definitions)} total field definitions")
            save_field_definitions(merged_definitions, args.file)
        else:
            # Parse and save directly
            parsed_definitions = parse_word_document(args.docx, args.start, args.end)
            if parsed_definitions:
                save_field_definitions(parsed_definitions, args.file)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 