import streamlit as st
import time
import json
import io
import os 
import requests
import base64
import pandas as pd
import difflib 
from datetime import date
from typing import Dict, Any, List, Optional
from PIL import Image, ImageEnhance, ExifTags 

# --- 0. Configuration and Initialization ---

# Set Streamlit page configuration before any other function is called
st.set_page_config(
    page_title="🛂 Unified KYC Document Extractor",
    layout="wide"
)

# Define the set of available document modes
DOCUMENT_MODES = {
    'Driving_License': 'Myanmar Driving License',
    'NRC': 'Myanmar National Registration Card (NRC)',
    'Passport': 'Myanmar Passport'
}

# --- API Configuration ---
try:
    # CRITICAL: Use st.secrets for secure API key loading
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback: Check environment variables for local testing
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

# The URL for the Gemini API model endpoint
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    
# --- Session State Initialization ---
# Initialize session state. Set Driving License as the default mode.
if 'document_mode' not in st.session_state: st.session_state['document_mode'] = 'Driving_License'
if 'extracted_data' not in st.session_state: st.session_state['extracted_data'] = None
if 'original_data' not in st.session_state: st.session_state['original_data'] = None
if 'accuracy_score' not in st.session_state: st.session_state['accuracy_score'] = None
if 'enhanced_image_bytes' not in st.session_state: st.session_state['enhanced_image_bytes'] = None
if 'uploaded_file_bytes' not in st.session_state: st.session_state['uploaded_file_bytes'] = None
if 'current_image_bytes' not in st.session_state: st.session_state['current_image_bytes'] = None
if 'uploaded_file_name' not in st.session_state: st.session_state['uploaded_file_name'] = None
if 'current_document_selection' not in st.session_state: st.session_state['current_document_selection'] = 'Driving_License' # Default selected mode


def reset_session_state_for_new_mode():
    """Clears all previous extraction data when the document mode is switched."""
    # Only reset if the actual mode has changed
    if st.session_state['document_mode'] != st.session_state['current_document_selection']:
        st.session_state['extracted_data'] = None
        st.session_state['original_data'] = None
        st.session_state['accuracy_score'] = None
        st.session_state['enhanced_image_bytes'] = None
        st.session_state['uploaded_file_bytes'] = None
        st.session_state['current_image_bytes'] = None
        st.session_state['uploaded_file_name'] = None
        # Update the document mode tracker
        st.session_state['document_mode'] = st.session_state['current_document_selection']
        # Force rerun to reflect state change immediately
        st.rerun()


# --- 1. Schemas and Prompts (Unified) ---
# --- NRC Schema and Prompt ---
NRC_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Overall_Confidence_Score": {"type": "NUMBER", "description": "A score from 0.0 to 1.0 indicating the model's certainty in the entire extraction."},
        "NRC_state_division": {"type": "STRING", "description": "The State/Division code (1-14), extracted as **standard Latin digits**."},
        "NRC_township": {"type": "STRING", "description": "The Township Code/initials, the Burmese words between '/' and '('. Extracted **precisely** in the original Burmese script."},
        "NRC_sth": {"type": "STRING", "description": "The classification code, the letters inside the parenthesis, e.g., '(N)', '(C)', or '(A)'. Must use standard Latin letters and include parentheses."},
        "NRC_no": {"type": "STRING", "description": "The six-digit citizen identification number, extracted as **standard Latin digits**."},
        "Name": {"type": "STRING", "description": "The cardholder's name (အမည်), extracted **precisely** in the original handwritten Burmese script."},
        "Fathers_Name": {"type": "STRING", "description": "The father's name (အဘအမည်), extracted **precisely** in the original handwritten Burmese script."},
        "Date_of_Birth": {"type": "STRING", "description": "The date of birth (မွေးသက္ကရာဇ်), extracted **precisely** in the original handwritten Burmese script (date, month, year)."}
    },
    "required": ["Overall_Confidence_Score", "NRC_state_division", "NRC_township", "NRC_sth", "NRC_no", "Name", "Fathers_Name", "Date_of_Birth"] 
}

NRC_SYSTEM_INSTRUCTION = (
    "You are an expert OCR and IDP system specialized in reading Myanmar National Registration Card (NRC) documents. "
    "Meticulously and accurately extract ONLY the core identity fields. Prioritize the **precise recognition of handwritten Burmese script** "
    "for Name, Fathers_Name, Date_of_Birth, and NRC_township. "
    "Ensure all numerical values are output as standard Latin digits (0-9). "
    "Output the results ONLY as a JSON object conforming to the provided schema."
)

NRC_USER_QUERY = (
    "Analyze the provided Myanmar NRC document image. Extract the values for ONLY the core identity fields: "
    "the four NRC components, Name, Father's Name, and Date of Birth. "
    "Crucially, split the NRC number (X/XXX(Y)######) into its four requested components: "
    "1. NRC_state_division (X, Latin digits 1-14) "
    "2. NRC_township (XXX, Burmese words between '/' and '(') "
    "3. NRC_sth ((Y), the classification code including parentheses, e.g., (N), (C), or (A)) "
    "4. NRC_no (######, the 6-digit number, Latin digits) "
    "Ensure Name, Father's Name, and Date of Birth are copied exactly as written in the original handwritten Burmese script. "
    "Return the output as a single JSON object."
)

# --- Driving License Schema and Prompt ---
DL_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "license_no": {"type": "string", "description": "The driving license number, typically like 'A/123456/22'."},
        "name": {"type": "string", "description": "The full name of the license holder in Latin script."},
        "nrc_no": {"type": "string", "description": "The NRC ID number, extracted exactly as seen on the card (usually Latin script, e.g., '12/MASANA(N)123456')."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "blood_type": {"type": "string", "description": "The blood type, e.g., 'A+', 'B', 'O-', 'AB'."},
        "valid_up": {"type": "string", "description": "The license expiry date in DD-MM-YYYY format."},
        "name_myanmar": {"type": "string", "description": "The full name of the license holder in Myanmar script (အမည်)."},
        "nrc_no_myanmar": {"type": "string", "description": "The NRC ID number fully converted/transliterated into Myanmar script (e.g., '၉/မထလ(နိုင်)၃၂၆၄၅၈')."},
        "date_of_birth_myanmar": {"type": "string", "description": "The date of birth in Myanmar script (မွေးသကရာဇ်)."},
        "valid_up_myanmar": {"type": "string", "description": "The license expiry date in Myanmar script (ကုန်ဆုံးရက်)."},
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
    },
    "required": ["license_no", "name", "nrc_no", "date_of_birth", "blood_type", "valid_up",
                  "name_myanmar", "nrc_no_myanmar", "date_of_birth_myanmar", "valid_up_myanmar", "extraction_confidence"]
}

DL_EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Driving License.
Extract ALL data fields, including both the Latin script (English) and Myanmar script (Burmese) values, and return the result strictly as a JSON object matching the provided schema.

CRITICAL INSTRUCTION FOR NRC:
1. 'nrc_no': Extract the NRC number **EXACTLY** as it appears on the card (e.g., '9/MAHTALA(N)326458').
2. 'nrc_no_myanmar': Transliterate the NRC number extracted in step 1 into **FULL Myanmar script** (e.g., '၉/မထလ(နိုင်)၃၂၆၄၅၈'). 

Ensure all Latin dates are in the DD-MM-YYYY format.
Finally, provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.
Do not include any extra text or formatting outside of the JSON object.
"""

# --- Passport Schema and Prompt ---
PP_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "description": "The passport type, e.g., 'PV' (Private)."},
        "country_code": {"type": "string", "description": "The country code, e.g., 'MMR'."},
        "passport_no": {"type": "string", "description": "The passport number (e.g., MH000000)."},
        "name": {"type": "string", "description": "The full name of the passport holder in Latin script (e.g., MIN ZAW)."},
        "nationality": {"type": "string", "description": "The nationality (e.g., MYANMAR)."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "sex": {"type": "string", "description": "The sex/gender, e.g., 'M' or 'F'."},
        "place_of_birth": {"type": "string", "description": "The place of birth (e.g., SAGAING)."},
        "date_of_issue": {"type": "string", "description": "The date of issue in DD-MM-YYYY format."},
        "date_of_expiry": {"type": "string", "description": "The date of expiry in DD-MM-YYYY format."},
        "authority": {"type": "string", "description": "The issuing authority (e.g., MOHA, YANGON)."},
        "mrz_full_string": {"type": "string", "description": "The two lines of the Machine Readable Zone (MRZ) combined into one string, separated by a space."},
        "passport_no_checksum": {"type": "string", "description": "The single checksum digit corresponding to the Passport No in the MRZ."},
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 to 1.0."}
    },
    "required": ["type", "country_code", "passport_no", "name", "nationality",
                  "date_of_birth", "sex", "place_of_birth", "date_of_issue",
                  "date_of_expiry", "authority", "mrz_full_string",
                  "passport_no_checksum", "extraction_confidence"]
}

PP_EXTRACTION_PROMPT = """
Analyze the provided image, which is a Passport (Biographical Data Page).
Extract ALL data fields shown on the page and the Machine Readable Zone (MRZ).

Return the result strictly as a JSON object matching the provided schema.
1. **Date Format**: Ensure all dates are converted to the **DD-MM-YYYY** format (e.g., 17 JAN 2023 -> 17-01-2023).
2. **MRZ**: Extract the two full lines of the Machine Readable Zone (MRZ) and combine them into a single string. Separate the two lines with a single space.
3. **Checksum**: Specifically extract the single digit checksum for the Passport No.
4. **Confidence**: Provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.

Do not include any extra text or formatting outside of the JSON object.
"""

# --- 2. Shared Image Processing & Utility Functions ---

MRZ_CHAR_VALUES = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '<': 0, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
    'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26,
    'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}
WEIGHTS = [7, 3, 1]

def calculate_mrz_checksum(data_string: str) -> str:
    """Calculates the checksum digit for a given MRZ data field using the Modulo 10 algorithm."""
    total_sum = 0
    for i, char in enumerate(data_string.upper()):
        value = MRZ_CHAR_VALUES.get(char, 0)
        weight = WEIGHTS[i % 3]
        total_sum += value * weight
    checksum = total_sum % 10
    return str(checksum)
    
def rotate_image_from_exif(img: Image.Image) -> Image.Image:
    """Reads EXIF data to automatically correct image orientation."""
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            orientation = exif.get(orientation_tag)
            
            if orientation == 3: img = img.rotate(180, expand=True)
            elif orientation == 6: img = img.rotate(-90, expand=True)
            elif orientation == 8: img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Converts a PIL Image object back to bytes."""
    buffer = io.BytesIO()
    # Use JPEG for display bytes to reduce memory, PNG for enhancement quality
    save_format = format if format == "PNG" else "JPEG" 
    img.save(buffer, format=save_format) 
    return buffer.getvalue()

def process_image(image_bytes: bytes, document_type: str) -> bytes:
    """Applies auto-rotation, manual rotation, and conditional enhancement."""
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Auto-Rotate from EXIF data
        img = rotate_image_from_exif(img)
        
        # Store the current display image bytes (after rotation)
        st.session_state['current_image_bytes'] = image_to_bytes(img, format="JPEG") 

        # 2. Conditional Enhancement (Only for NRC due to focus on handwriting)
        if document_type == 'NRC':
            # Convert to grayscale for better contrast/sharpness focusing on text
            img_gray = img.convert("L")
            sharpen_filter = ImageEnhance.Sharpness(img_gray)
            img_sharpened = sharpen_filter.enhance(2.0) 
            contrast_filter = ImageEnhance.Contrast(img_sharpened)
            img_final = contrast_filter.enhance(1.5) 
            # Use PNG for the AI input for maximum quality transmission
            final_bytes = image_to_bytes(img_final, format="PNG") 
        else:
            # DL/Passport: Use original color image but save as PNG for AI input
            final_bytes = image_to_bytes(img, format="PNG")
            
        return final_bytes
        
    except Exception as e:
        st.warning(f"Image processing pipeline failed: {e}. Using original image bytes.")
        return image_bytes

def rotate_uploaded_image(angle: int):
    """Handles user-triggered manual rotation."""
    if 'uploaded_file_bytes' not in st.session_state or st.session_state['uploaded_file_bytes'] is None:
        st.error("Please upload an image first.")
        return
        
    try:
        # Load the image from the latest display bytes
        img_bytes = st.session_state.get('current_image_bytes') or st.session_state['uploaded_file_bytes']
        img = Image.open(io.BytesIO(img_bytes))
        
        # Apply rotation
        img_rotated = img.rotate(angle, expand=True)
        
        # Save the new bytes back to session state
        st.session_state['uploaded_file_bytes'] = image_to_bytes(img_rotated, format="JPEG")
        st.session_state['current_image_bytes'] = st.session_state['uploaded_file_bytes']

        # Clear old extraction results
        st.session_state['extracted_data'] = None 
        st.session_state['accuracy_score'] = None
        st.session_state['enhanced_image_bytes'] = None 
        
        st.rerun()
    except Exception as e:
        st.error(f"Error during manual rotation: {e}")

# --- 3. NRC Specific Validation and Scoring (Functions remain the same) ---

def is_valid_date(burmese_date_str: str) -> bool:
    """Basic plausibility check for the Date of Birth using Latin digits if found."""
    today = date.today()
    import re
    # Try to find a 4-digit year (even within Burmese text)
    year_match = re.search(r'(\d{4})', burmese_date_str)
    if year_match:
        try:
            year = int(year_match.group(1))
            if year > today.year: return False
            if today.year - year > 150: return False 
            return True
        except ValueError:
            return False
    # If no Latin year is found, we assume it's correctly written in Burmese script
    return True 

def validate_nrc_data(data: Dict[str, Any]) -> List[str]:
    """Applies the NRC business rules."""
    warnings = []
    
    # 1. Validate NRC State/Division Code
    nrc_state_code = str(data.get('NRC_state_division', '')).strip()
    try:
        code = int(nrc_state_code)
        if not (1 <= code <= 14):
            warnings.append(f"State/Division Code '{nrc_state_code}' is outside the valid range (1-14).")
    except ValueError:
        if nrc_state_code:
            warnings.append(f"State/Division Code '{nrc_state_code}' is not a valid Latin number (1-14).")
    if not nrc_state_code: warnings.append("NRC State/Division Code is missing.")
    
    # 2. Validate NRC 6-digit Number
    nrc_no = str(data.get('NRC_no', '')).strip()
    if not (nrc_no.isdigit() and len(nrc_no) == 6):
        warnings.append(f"NRC 6-digit number '{nrc_no}' is invalid. It must contain exactly 6 Latin digits.")
    
    # 3. Validate NRC Status (sth)
    nrc_sth = str(data.get('NRC_sth', '')).strip()
    # Check if it looks like (A), (C), or (N) including parentheses
    if nrc_sth and not (nrc_sth.startswith('(') and nrc_sth.endswith(')') and len(nrc_sth) >= 3):
        warnings.append(f"NRC Classification Code ('NRC_sth') '{nrc_sth}' should be formatted as (C), (N), or (A) and include parentheses.")
    
    # 4. Validate Date of Birth
    dob_burmese = data.get('Date_of_Birth', '').strip()
    if dob_burmese and not is_valid_date(dob_burmese):
        warnings.append(f"Date of Birth appears suspicious (e.g., future date or over 150 years old).")
            
    return warnings

def calculate_accuracy_score(original_data: Dict[str, Any], corrected_data: Dict[str, Any]) -> float:
    """Calculates field-level string similarity (NRC only)."""
    fields_to_compare = [
        "NRC_state_division", "NRC_township", "NRC_sth", "NRC_no", 
        "Name", "Fathers_Name", "Date_of_Birth"
    ]
    total_score = 0.0
    
    for field in fields_to_compare:
        original_value = str(original_data.get(field, "")).strip()
        corrected_value = str(corrected_data.get(field, "")).strip()
        
        if original_value or corrected_value:
            similarity_ratio = difflib.SequenceMatcher(None, original_value, corrected_value).ratio()
            total_score += similarity_ratio
        else:
            total_score += 1.0 # Perfect match if both are empty/missing
            
    accuracy = total_score / len(fields_to_compare)
    return accuracy

def update_nrc_data_from_fields(fields_data: Dict[str, str]):
    """Updates session state with human-corrected NRC data."""
    updated_data = st.session_state['extracted_data'].copy() 
    for key, value in fields_data.items():
        # Update field values
        if key in updated_data:
             updated_data[key] = value

    st.session_state['extracted_data'] = updated_data
    
    # Re-validate the corrected data
    updated_data['validation_warnings'] = validate_nrc_data(updated_data)

    # Recalculate accuracy against the original model output
    original_data = st.session_state['original_data']
    accuracy = calculate_accuracy_score(original_data, updated_data)
    st.session_state['accuracy_score'] = accuracy
    
    st.success("NRC data successfully updated, re-validated, and accuracy score recalculated.")

# --- 4. Core AI Extraction Function ---

def extract_kyc_data(enhanced_image_bytes: bytes, document_type: str) -> Optional[Dict[str, Any]]:
    """Calls the Gemini API to extract structured data based on document type."""
    
    # Select specific configuration
    if document_type == 'NRC':
        schema = NRC_JSON_SCHEMA
        user_query = NRC_USER_QUERY
        system_instruction = NRC_SYSTEM_INSTRUCTION
    elif document_type == 'Driving_License':
        schema = DL_EXTRACTION_SCHEMA
        user_query = DL_EXTRACTION_PROMPT
        system_instruction = "You are an expert OCR system for extracting data from official documents. Follow all instructions precisely."
    elif document_type == 'Passport':
        schema = PP_EXTRACTION_SCHEMA
        user_query = PP_EXTRACTION_PROMPT
        system_instruction = "You are an expert OCR system for extracting data from official documents, especially focusing on biographical data and Machine Readable Zones (MRZ)."
    else:
        st.error("Invalid document type selected.")
        return None
        
    base64_image = base64.b64encode(enhanced_image_bytes).decode('utf-8')
    mime_type = "image/png" # Use PNG for the AI input for quality
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_query},
                    {"inlineData": {"mimeType": mime_type, "data": base64_image}}
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.0 # Use 0.0 for deterministic extraction
        }
    }

    extraction_bar = st.progress(0, text="Sending image to Gemini for extraction...")
    
    try:
        # Mimic exponential backoff manually for a single retry attempt
        for attempt in range(2):
            try:
                response = requests.post(
                    API_URL, 
                    params={'key': API_KEY},
                    headers={'Content-Type': 'application/json'}, 
                    json=payload
                )
                response.raise_for_status()
                extraction_bar.progress(100, text="Extraction complete!")
                time.sleep(0.5)
                break
            except requests.exceptions.RequestException as e:
                if attempt == 0 and response.status_code in [429, 503]:
                    time.sleep(2 ** (attempt + 1)) # Exponential backoff
                    extraction_bar.progress(50, text="API throttled, retrying...")
                    continue
                raise e # Re-raise if not a retryable error or last attempt

        result = response.json()
        
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            extracted_data = json.loads(json_string)
            st.session_state['original_data'] = extracted_data.copy()
            st.session_state['document_mode'] = document_type
            
            # Run specific post-processing/validation
            if document_type == 'NRC':
                warnings = validate_nrc_data(extracted_data)
                extracted_data['validation_warnings'] = warnings
            
            return extracted_data
        else:
            error_detail = result.get('candidates', [{}])[0].get('finishReason', 'No content generated.')
            st.error(f"Error: Could not extract structured JSON. Finish Reason: {error_detail}")
            return None

    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None
    finally:
        extraction_bar.empty()

# --- 5. UI Rendering Functions ---

def render_passport_results(data: Dict[str, Any]):
    """Renders passport results and runs MRZ checksum validation."""
    
    st.header("Results: Passport Data Extraction")
    
    passport_no_data = data.get('passport_no', '').replace('<', '')
    extracted_checksum = data.get('passport_no_checksum', '')
    calculated_checksum = calculate_mrz_checksum(passport_no_data)
    
    checksum_verified = (calculated_checksum == extracted_checksum) and (extracted_checksum != "")
    verification_status = "✅ VERIFIED (Checksum Matched)" if checksum_verified else "⚠️ WARNING: CHECKSUM MISMATCH"
    
    st.subheader("Verification Status")
    if checksum_verified:
        st.success(verification_status)
    else:
        st.warning(verification_status)
        st.error(f"Extracted Checksum: **{extracted_checksum}** | Calculated Checksum: **{calculated_checksum}**")

    # Display in a simple table/list
    data_view = [
        ("Extraction Confidence", f"{data.get('extraction_confidence', 0.0):.2f}"),
        ("--- BIOGRAPHICAL DATA ---", ""),
        ("Passport Type", data.get("type", "N/A")),
        ("Country Code", data.get("country_code", "N/A")),
        ("Passport No", data.get("passport_no", "N/A")),
        ("Name", data.get("name", "N/A")),
        ("Nationality", data.get("nationality", "N/A")),
        ("Date of Birth", data.get("date_of_birth", "N/A")),
        ("Sex", data.get("sex", "N/A")),
        ("Place of Birth", data.get("place_of_birth", "N/A")),
        ("Date of Issue", data.get("date_of_issue", "N/A")),
        ("Date of Expiry", data.get("date_of_expiry", "N/A")),
        ("Authority", data.get("authority", "N/A")),
        ("--- MRZ DATA ---", ""),
        ("MRZ Full String", data.get("mrz_full_string", "N/A")),
        ("Passport No Checksum", extracted_checksum),
        ("Calculated Checksum", calculated_checksum),
    ]
    st.table(pd.DataFrame(data_view, columns=['Field', 'Value']))
    

def render_dl_results(data: Dict[str, Any]):
    """Renders Driving License results."""
    
    st.header("Results: Driving License Data Extraction")
    
    confidence = data.get('extraction_confidence', 0.0)
    st.metric(
        label="Model Confidence",
        value=f"{confidence * 100:.0f}%",
        delta="AI's certainty of its initial output."
    )
    
    # Display in a two-column format for English and Myanmar scripts
    col_en, col_my = st.columns(2)
    
    with col_en:
        st.subheader("Latin Script (English) Fields")
        st.table([
            ("License No", data.get("license_no", "N/A")),
            ("Name", data.get("name", "N/A")),
            ("NRC No (Latin)", data.get("nrc_no", "N/A")),
            ("Date of Birth", data.get("date_of_birth", "N/A")),
            ("Valid Up", data.get("valid_up", "N/A")),
        ])
    with col_my:
        st.subheader("Myanmar Script (Burmese) Fields")
        st.table([
            ("အမည် (Name)", data.get("name_myanmar", "N/A")),
            ("မှတ်ပုံတင် (NRC)", data.get("nrc_no_myanmar", "N/A")),
            ("မွေးသက္ကရာဇ် (DOB)", data.get("date_of_birth_myanmar", "N/A")),
            ("ကုန်ဆုံးရက် (Expiry)", data.get("valid_up_myanmar", "N/A")),
            ("Blood Type", data.get("blood_type", "N/A")),
        ])

def render_nrc_results(data: Dict[str, Any]):
    """Renders NRC results, validation, and the correction form (HITL)."""
    
    st.header("Results: NRC Data Extraction")
    
    warnings = data.get('validation_warnings', [])
    confidence = data.get('Overall_Confidence_Score')
    accuracy = st.session_state.get('accuracy_score', 1.0)
    
    # --- Performance Metrics Section ---
    col_conf, col_acc = st.columns(2)
    
    with col_conf:
        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.0f}%" if confidence is not None else "N/A"
        )
    with col_acc:
        st.metric(
            label="OCR Field Accuracy",
            value=f"{accuracy * 100:.2f}%",
            delta="Similarity to the human-corrected output."
        )

    # 1. Validation Warnings
    st.subheader("⚠️ 1. Validation Warnings")
    if warnings:
        st.warning("The extracted data has the following potential errors:")
        for warning in warnings:
            st.markdown(f"- **{warning}**")
    else:
        st.success("Data passed all preliminary validation checks.")
    
    st.markdown("---")

    # 2. Extracted Data Snapshot
    st.subheader("✅ 2. Extracted Data Snapshot")
    final_data_view = [
        ("NRC State/Division (X)", data.get("NRC_state_division", "N/A")),
        ("NRC Township (XXX)", data.get("NRC_township", "N/A")),
        ("NRC Classification ((Y))", data.get("NRC_sth", "N/A")),
        ("NRC 6-Digit No (######)", data.get("NRC_no", "N/A")),
        ("Name (အမည်)", data.get("Name", "N/A")),
        ("Father's Name (အဘအမည်)", data.get("Fathers_Name", "N/A")),
        ("Date of Birth (မွေးသက္ကရာဇ်)", data.get("Date_of_Birth", "N/A")),
    ]
    st.table(pd.DataFrame(final_data_view, columns=['Field', 'Value']))
    
    st.markdown("---")

    # 3. Human-in-the-Loop Correction (Text Fields)
    st.subheader("✍️ 3. Correct Data & Recalculate Accuracy (HITL)")
    st.markdown("Review and correct any errors below to train the accuracy score. Note: NRC_township, Name, Father's Name, and Date of Birth require **Burmese script**.")
    
    with st.form("nrc_correction_form"):
        current_data = st.session_state['extracted_data']
        
        # --- Granular NRC Fields ---
        st.markdown("##### NRC Components (X/XXX(Y)######)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            NRC_state_division = st.text_input("State/Division Code (X)", value=current_data.get('NRC_state_division', ''), help="Latin Digits (1-14)")
        with col2:
            NRC_township = st.text_input("Township Code (XXX)", value=current_data.get('NRC_township', ''), help="Burmese Script Only")
        with col3:
            NRC_sth = st.text_input("Classification Code ((Y))", value=current_data.get('NRC_sth', ''), help="(N), (C), or (A) including parentheses")
        with col4:
            NRC_no = st.text_input("6-Digit Number (######)", value=current_data.get('NRC_no', ''), help="Latin Digits Only")

        st.markdown("---")

        # --- Personal Details (Burmese Script) ---
        Name = st.text_input("Name (အမည်)", value=current_data.get('Name', ''), help="Burmese Script Only")
        Fathers_Name = st.text_input("Father's Name (အဘအမည်)", value=current_data.get('Fathers_Name', ''), help="Burmese Script Only")
        Date_of_Birth = st.text_input("Date of Birth (မွေးသက္ကရာဇ်)", value=current_data.get('Date_of_Birth', ''), help="Burmese Script Only")

        # Button to submit corrections
        submitted = st.form_submit_button("Update and Re-Validate Extracted Data", type="secondary")
        
        if submitted:
            fields_data = {
                "NRC_state_division": NRC_state_division,
                "NRC_township": NRC_township,
                "NRC_sth": NRC_sth,
                "NRC_no": NRC_no,
                "Name": Name,
                "Fathers_Name": Fathers_Name,
                "Date_of_Birth": Date_of_Birth,
            }
            update_nrc_data_from_fields(fields_data)
            st.rerun()

# --- 6. Main Streamlit Application Logic ---

def main_app():
    st.title("🛂 Unified KYC Document Extractor")
    st.caption("Leveraging Gemini 2.5 Flash for structured data extraction from Myanmar IDs.")

    # --- Sidebar for selection and Upload ---
    st.sidebar.header("Configuration")
    
    # Selectbox for document type (limited to the three modes)
    selected_mode = st.sidebar.selectbox(
        "Select Document Type",
        options=list(DOCUMENT_MODES.keys()),
        format_func=lambda x: DOCUMENT_MODES[x],
        key='current_document_selection',
        on_change=reset_session_state_for_new_mode,
        help="Select the type of document to be processed."
    )
    
    # Check for mode change before processing upload
    if st.session_state['document_mode'] != selected_mode:
        st.session_state['document_mode'] = selected_mode # Update the mode tracker if reset_session_state_for_new_mode hasn't rerun yet
    
    current_doc_type = st.session_state['document_mode']
    
    uploaded_file = st.sidebar.file_uploader(
        f"Upload {DOCUMENT_MODES.get(current_doc_type, 'Document')}", 
        type=['png', 'jpg', 'jpeg'],
        key='uploaded_file_ui'
    )
    
    # --- File Upload Handling and Image Processing ---
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get('uploaded_file_name'):
            # New file uploaded or mode changed, read new bytes
            st.session_state['uploaded_file_bytes'] = uploaded_file.getvalue()
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.session_state['extracted_data'] = None # Reset data on new upload
            st.session_state['accuracy_score'] = None
            st.session_state['current_image_bytes'] = None # Clear display image to trigger processing

        # Only process if image bytes are available
        if st.session_state['current_image_bytes'] is None and st.session_state['uploaded_file_bytes']:
            # Run initial processing (EXIF rotation + conditional enhancement)
            with st.spinner(f"Processing image for {DOCUMENT_MODES[current_doc_type]}..."):
                enhanced_bytes = process_image(st.session_state['uploaded_file_bytes'], current_doc_type)
                st.session_state['enhanced_image_bytes'] = enhanced_bytes
            
            # Note: A rerun is triggered by rotation changes, so we rely on session state after this block.

        # --- Layout for Image and Results ---
        col_img, col_results = st.columns([1, 1.5])

        with col_img:
            st.subheader("Original Image & Controls")
            
            # Display the current image (after rotation/EXIF fix)
            display_bytes = st.session_state.get('current_image_bytes', st.session_state['uploaded_file_bytes'])
            st.image(display_bytes, caption=uploaded_file.name, use_column_width=True)
            
            # Rotation Controls
            rot_col1, rot_col2 = st.columns(2)
            with rot_col1:
                st.button("Rotate Left  counter-clockwise", on_click=rotate_uploaded_image, args=(-90,), use_container_width=True)
            with rot_col2:
                st.button("Rotate Right clockwise", on_click=rotate_uploaded_image, args=(90,), use_container_width=True)

            if current_doc_type == 'NRC' and st.session_state.get('enhanced_image_bytes'):
                with st.expander("View Enhanced Image (for AI input)"):
                    st.image(st.session_state['enhanced_image_bytes'], caption="Sharpened/Contrast-Adjusted Image (Grayscale)", use_column_width=True)

        with col_results:
            st.subheader(f"Extraction Results for {DOCUMENT_MODES[current_doc_type]}")
            
            # --- Extraction Trigger ---
            if st.session_state['extracted_data'] is None:
                if st.button(f"Start Structured Extraction", type="primary", use_container_width=True):
                    with st.spinner(f"Requesting data extraction for {current_doc_type} via Gemini..."):
                        extracted_data = extract_kyc_data(st.session_state['enhanced_image_bytes'], current_doc_type)
                        if extracted_data:
                            st.session_state['extracted_data'] = extracted_data
                            # For NRC, initialize accuracy to 1.0 (perfect before human correction)
                            if current_doc_type == 'NRC':
                                st.session_state['accuracy_score'] = 1.0
                            st.rerun() # Rerun to display results
                else:
                    st.info("Click 'Start Structured Extraction' to analyze the image.")
            
            # --- Results Rendering ---
            if st.session_state['extracted_data']:
                if current_doc_type == 'NRC':
                    render_nrc_results(st.session_state['extracted_data'])
                elif current_doc_type == 'Driving_License':
                    render_dl_results(st.session_state['extracted_data'])
                elif current_doc_type == 'Passport':
                    render_passport_results(st.session_state['extracted_data'])

    else:
        st.info(f"Please use the sidebar to select the document type ({DOCUMENT_MODES[current_doc_type]} selected) and upload an image file to begin.")


if __name__ == "__main__":
    main_app()
