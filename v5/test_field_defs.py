#!/usr/bin/env python3
"""
Test script for field definitions
"""

import json
import sys
import re

# Path to field definitions file
FIELD_DEFINITIONS_FILE = "field_definitions.json"

# Load field definitions from a JSON file
def load_field_definitions(file_path=FIELD_DEFINITIONS_FILE):
    """Load field definitions from a JSON file"""
    try:
        if file_path:
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}
    except Exception as e:
        print(f"Error loading field definitions: {e}")
        return {}

# Clean up field name for matching
def clean_field_name(field_name):
    """Clean up field name for better matching"""
    if not field_name:
        return ""
    
    # Convert to lowercase and strip whitespace
    field_name = field_name.lower().strip()
    
    # Remove prefixes like "explain", "what is", etc.
    prefixes_to_remove = ["explain ", "what is ", "tell me about ", "describe "]
    for prefix in prefixes_to_remove:
        if field_name.startswith(prefix):
            field_name = field_name[len(prefix):].strip()
            break
    
    # Remove trailing punctuation
    if field_name.endswith((':', '?', '.')):
        field_name = field_name[:-1].strip()
    
    return field_name

# Find field by name
def find_field_by_name(field_definitions, field_name):
    """Find a field ID by name or description"""
    cleaned_name = clean_field_name(field_name)
    
    # Direct field ID lookup
    if cleaned_name.isdigit() and cleaned_name in field_definitions:
        return cleaned_name
    
    # Look for exact title match
    for field_id, field_data in field_definitions.items():
        title = field_data.get("title", "").lower().rstrip(':')
        if cleaned_name == title:
            return field_id
    
    # Look for partial match
    for field_id, field_data in field_definitions.items():
        title = field_data.get("title", "").lower().rstrip(':')
        if cleaned_name in title:
            return field_id
    
    return None

# Helper function to format field definitions
def format_field_definition(field_definitions, field_id, with_line_breaks=True):
    """Format a field definition with either line breaks or dashes"""
    field_data = field_definitions.get(field_id)
    if not field_data:
        return f"Field {field_id} not found."
        
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
        return f"""**{field_id}. {title}**
**Format:** {fmt}
**Technical Name:** {tech_name}
**Length:** {length}
**Positions:** {positions}
**Definition:** {definition}"""
    else:
        return f"**{field_id}. {title}** - **Format:** {fmt} - **Technical Name:** {tech_name} - **Length:** {length} - **Positions:** {positions} - **Definition:** {definition}"

def main():
    field_definitions = load_field_definitions()
    print(f"Loaded {len(field_definitions)} field definitions.")
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
        # Check if it's a direct field ID
        if query.isdigit() and query in field_definitions:
            field_id = query
        else:
            # Try to find field by name
            field_id = find_field_by_name(field_definitions, query)
        
        if field_id:
            print(f"Found field ID: {field_id}")
            print(format_field_definition(field_definitions, field_id, True))
        else:
            print(f"No matching field found for: {query}")
    else:
        # Show available fields
        print("Available fields:")
        for field_id, field_data in sorted(field_definitions.items(), key=lambda x: int(x[0])):
            title = field_data.get("title", "")
            print(f"{field_id}: {title}")
        
        print("\nRun with field ID or name as argument, e.g.: python test_field_defs.py 141")
        print("Or: python test_field_defs.py \"explain Claim-Level ICD Procedure Code 1\"")

if __name__ == "__main__":
    main() 