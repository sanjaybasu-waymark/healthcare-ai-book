"""
Chapter 29 - Example 5
Extracted from Healthcare AI Implementation Guide
"""

\# Excerpt from sustainable_ai_healthcare_code.py
\# --- 5. Comprehensive Error Handling (Illustrative Examples) ---

def process_patient_record(record_data):
    try:
        \# Simulate data validation
        if not isinstance(record_data, dict):
            raise TypeError("Record data must be a dictionary.")
        if "patient_id" not in record_data or not isinstance(record_data["patient_id"], str):
            raise ValueError("Missing or invalid patient_id.")
        if "temperature" in record_data and not (35.0 <= record_data["temperature"] <= 42.0):
            raise ValueError("Temperature out of clinical range.")

        \# Simulate a critical operation that might fail
        if record_data.get("simulate_db_error"):
            raise ConnectionError("Database connection failed.")

        print(f"\nSuccessfully processed patient record for ID: {record_data["patient_id"]}")
        return {"status": "success", "patient_id": record_data["patient_id"]}

    except TypeError as e:
        print(f"Error: Invalid data type for patient record. Details: {e}")
        return {"status": "failed", "error": str(e), "type": "data_type_error"}
    except ValueError as e:
        print(f"Error: Data validation failed for patient record. Details: {e}")
        return {"status": "failed", "error": str(e), "type": "validation_error"}
    except ConnectionError as e:
        print(f"Error: Critical system failure during patient record processing. Details: {e}")
        return {"status": "failed", "error": str(e), "type": "system_error"}
    except Exception as e:
        print(f"An unexpected error occurred during patient record processing: {e}")
        return {"status": "failed", "error": str(e), "type": "unexpected_error"}