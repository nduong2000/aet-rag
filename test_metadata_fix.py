#!/usr/bin/env python3

import json

# Simulate the ChromaDB metadata as it's stored (with JSON strings)
test_metadata = {
    "source_file": "test-doc.pdf", 
    "field_numbers": '["15", "16", "17"]',  # This is stored as a JSON string
    "document_types": '["universal_medical_file", "field_definitions"]',
    "contains_field_definitions": True,
    "other_field": "normal_string"
}

def _deserialize_metadata(metadata):
    """Test version of the deserialization function"""
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
                    print(f"✓ Deserialized {field}: {parsed_value}")
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, leave as string and log warning
                print(f"⚠ Failed to deserialize {field}: {deserialized[field]}")
    
    return deserialized

def test_frontend_processing(field_numbers):
    """Test the frontend array processing"""
    print(f"\n🧪 Testing frontend processing with field_numbers: {field_numbers}")
    print(f"   Type: {type(field_numbers)}")
    print(f"   Is array: {isinstance(field_numbers, list)}")
    
    if isinstance(field_numbers, list):
        try:
            # This is what the frontend JavaScript does
            field_display = field_numbers[:3]  # slice
            field_text = ", ".join(str(f) for f in field_display)  # join
            print(f"   ✅ SUCCESS: {field_text}")
            return True
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    else:
        print(f"   ❌ ERROR: field_numbers is not an array!")
        return False

if __name__ == "__main__":
    print("🔧 Testing Metadata Deserialization Fix")
    print("=" * 50)
    
    print("\n📋 Original metadata (as stored in ChromaDB):")
    for key, value in test_metadata.items():
        print(f"   {key}: {value} ({type(value).__name__})")
    
    print("\n🔄 After deserialization:")
    fixed_metadata = _deserialize_metadata(test_metadata)
    for key, value in fixed_metadata.items():
        print(f"   {key}: {value} ({type(value).__name__})")
    
    print("\n🧪 Testing frontend compatibility:")
    # Test with original (broken) data
    test_frontend_processing(test_metadata['field_numbers'])
    
    # Test with fixed data  
    test_frontend_processing(fixed_metadata['field_numbers'])
    
    print("\n" + "=" * 50)
    print("🎯 If both tests show SUCCESS, the fix should work!") 