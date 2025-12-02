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
from PIL import Image, ImageEnhance, ExifTags # Import PIL components

# --- 0. Configuration and Initialization ---

st.set_page_config(
    page_title="🛂 Unified KYC Document Extractor (AI IDP)",
    layout="wide"
)

# --- API Configuration ---
try:
    # 💥 CRITICAL: Use st.secrets for secure API key loading
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback: Check environment variables for local testing
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets or environment variables. Please configure your key.")
    st.stop()
    
# The URL for the Gemini API model endpoint
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    
# --- Session State Initialization ---
# Initialize session state for all shared data across document types
if 'extracted_data' not in st.session_state: st.session_state['extracted_data'] = None
if 'original_data' not in st.session_state: st.session_state['original_data'] = None
if 'accuracy_score' not in st.session_state: st.session_state['accuracy_score'] = None
if 'enhanced_image_bytes' not in st.session_state: st.session_state['enhanced_image_bytes'] = None
if 'uploaded_file_bytes' not in st.session_state: st.session_state['uploaded_file_bytes'] = None
if 'current_image_bytes' not in st.session_state: st.session_state['current_image_bytes'] = None
if 'uploaded_file_name' not in st.session_state: st.session_state['uploaded_file_name'] = None

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
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
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
    img.save(buffer, format=format) 
    return buffer.getvalue()

def process_image(image_bytes: bytes, rotation_angle: int = 0, document_type: str = 'NRC') -> bytes:
    """Applies auto-rotation, manual rotation, and conditional enhancement."""
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Auto-Rotate from EXIF data
        img = rotate_image_from_exif(img)
        
        # 2. Apply Manual Rotation
        if rotation_angle != 0:
            img = img.rotate(rotation_angle, expand=True)

        # Store the manually rotated image bytes for display (before enhancement)
        st.session_state['current_image_bytes'] = image_to_bytes(img, format="JPEG") # Display in JPEG for smaller size

        # 3. Conditional Enhancement (Only for NRC due to focus on handwriting)
        if document_type == 'NRC':
            # NRC Enhancement Pipeline (Sharpen 2.0x, Contrast 1.5x)
            img_gray = img.convert("L")
            sharpen_filter = ImageEnhance.Sharpness(img_gray)
            img_sharpened = sharpen_filter.enhance(2.0) 
            contrast_filter = ImageEnhance.Contrast(img_sharpened)
            img_final = contrast_filter.enhance(1.5) 
            final_bytes = image_to_bytes(img_final, format="PNG") # Use PNG for quality retention
            
        else:
            # DL/Passport: Use original color image but save as PNG for quality
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
        # Load the image from the latest bytes (which might already be rotated/processed)
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

# --- 3. NRC Specific Validation and Scoring ---

def is_valid_date(burmese_date_str: str) -> bool:
    """Basic plausibility check for the Date of Birth using Latin digits if found."""
    today = date.today()
    import re
    year_match = re.search(r'(\d{4})', burmese_date_str)
    if year_match:
        try:
            year = int(year_match.group(1))
            if year > today.year: return False
            if today.year - year > 150: return False 
            return True
        except ValueError:
            return False
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
            total_score += 1.0 
            
    accuracy = total_score / len(fields_to_compare)
    return accuracy

def update_nrc_data_from_fields(fields_data: Dict[str, str]):
    """Updates session state with human-corrected NRC data."""
    updated_data = st.session_state['extracted_data'].copy() 
    for key, value in fields_data.items():
        # Ensure 'Overall_Confidence_Score' is not overwritten by text input
        if key not in ['Overall_Confidence_Score_input']: 
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

@st.cache_data(show_spinner=False)
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
    mime_type = "image/png" # Using PNG for consistent quality from the pre-processor
    
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
            "temperature": 0.0 # Low temperature for deterministic data extraction
        }
    }

    try:
        response = requests.post(
            API_URL, 
            params={'key': API_KEY},
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            extracted_data = json.loads(json_string)
            st.session_state['original_data'] = extracted_data.copy()
            
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

# --- 5. UI Rendering Functions ---

def render_passport_results(data: Dict[str, Any]):
    """Renders passport results and runs MRZ checksum validation."""
    
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
        ("Passport Type", data.get("type", "N/A")),
        ("Passport No", data.get("passport_no", "N/A")),
        ("Name", data.get("name", "N/A")),
        ("Date of Birth", data.get("date_of_birth", "N/A")),
        ("Date of Expiry", data.get("date_of_expiry", "N/A")),
        ("MRZ Full String", data.get("mrz_full_string", "N/A")),
        ("Extraction Confidence", f"{data.get('extraction_confidence', 0.0):.2f}"),
    ]
    st.table(data_view)
    

def render_dl_results(data: Dict[str, Any]):
    """Renders Driving License results."""
    
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
    
    warnings = data.get('validation_warnings', [])
    confidence = data.get('Overall_Confidence_Score')
    accuracy = st.session_state.get('accuracy_score', 1.0)
    
    # --- Performance Metrics Section ---
    st.header("✨ NRC Performance Metrics")
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

    # 2. Final Data Display
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
    st.table(final_data_view)
    
    st.markdown("---")

    # 3. Human-in-the-Loop Correction (Text Fields)
    st.subheader("✍️ 3. Correct Data & Recalculate Accuracy (HITL)")
    st.markdown("Review and correct any errors below to train the accuracy score.")
    
    with st.form("nrc_correction_form"):
        current_data = st.session_state['extracted_data']
        
        # --- Granular NRC Fields ---
        st.markdown("##### NRC Components (X/XXX(Y)######)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            NRC_state_division = st.text_input("State/Division Code (X)", value=current_data.get('NRC_state_division', ''))
        with col2:
            NRC_township = st.text_input("Township Code (XXX)", value=current_data.get('NRC_township', ''))
        with col3:
            NRC_sth = st.text_input("Classification Code ((Y))", value=current_data.get('NRC_sth', ''))
        with col4:
            NRC_no = st.text_input("6-Digit Number (######)", value=current_data.get('NRC_no', ''))

        st.markdown("---")

        # --- Personal Details (Burmese Script) ---
        Name = st.text_input("Name (အမည်)", value=current_data.get('Name', ''))
        Fathers_Name = st.text_input("Father's Name (အဘအမည်)", value=current_data.get('Fathers_Name', ''))
        Date_of_Birth = st.text_input("Date of Birth (မွေးသက္ကရာဇ်)", value=current_data.get('Date_of_Birth', ''))

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

# --- 6. Main Application Flow ---

def main():
    st.title("🛂 Unified KYC Document Extractor (AI IDP)")
    st.caption("Extract structured data from Myanmar NRC, Driving License, and Passports using the Gemini API.")

    # --- Document Type Selector ---
    document_type_options = ["NRC", "Driving_License", "Passport"]
    selected_document_type = st.sidebar.selectbox(
        "Select Document Type for Scanning",
        document_type_options,
        index=0,
        format_func=lambda x: x.replace('_', ' ')
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Mode: {selected_document_type.replace('_', ' ')}**")
    
    # --- File/Camera Input ---
    tab1, tab2 = st.tabs(["🖼️ Upload Image", "📸 Take Photo"])

    uploaded_file = None
    
    with tab1:
        uploaded_file_widget = st.file_uploader(f"Upload {selected_document_type.replace('_', ' ')} Image", type=["png", "jpg", "jpeg"])
    with tab2:
        camera_image_widget = st.camera_input(f"Take a Photo of {selected_document_type.replace('_', ' ')}")

    if uploaded_file_widget:
        uploaded_file = uploaded_file_widget
    elif camera_image_widget:
        uploaded_file = camera_image_widget
    
    
    if uploaded_file:
        file_name_or_default = getattr(uploaded_file, 'name', 'camera_image_file')

        # Only update session state bytes if a new file/photo is detected
        if st.session_state.get('uploaded_file_name') != file_name_or_default or st.session_state.get('uploaded_file_bytes') is None:
            st.session_state['uploaded_file_bytes'] = uploaded_file.getvalue()
            st.session_state['uploaded_file_name'] = file_name_or_default
            st.session_state['current_image_bytes'] = st.session_state['uploaded_file_bytes'] # Reset current image on new upload/capture
            st.session_state['extracted_data'] = None
            st.session_state['accuracy_score'] = None
            st.session_state['enhanced_image_bytes'] = None
            st.rerun()

    
    if st.session_state.get('uploaded_file_bytes'):
        image_bytes_to_display = st.session_state['current_image_bytes']
        
        # --- Rotation Controls ---
        st.subheader("0. Correct Orientation (If needed)")
        col_rot1, col_rot2, col_rot3, col_rot4 = st.columns([1, 1, 1, 3])
        with col_rot1: st.button("Rotate ↺ -90°", on_click=rotate_uploaded_image, args=(90,), help="Rotate Counter-Clockwise 90°")
        with col_rot2: st.button("Rotate ↻ +90°", on_click=rotate_uploaded_image, args=(-90,), help="Rotate Clockwise 90°")
        with col_rot3: st.button("Rotate 180°", on_click=rotate_uploaded_image, args=(180,), help="Rotate 180°")
        with col_rot4: st.info("The image below shows the *current* orientation (after auto-EXIF correction).")

        # --- Image Display Tabs ---
        img_tab1, img_tab2 = st.tabs(["Current Oriented Image", "AI Processed Image (Enhanced)"])
        
        with img_tab1:
            st.image(image_bytes_to_display, caption='Current Input Document', use_column_width=True)

        with img_tab2:
            if st.session_state.get('enhanced_image_bytes') is not None:
                st.image(st.session_state['enhanced_image_bytes'], caption='AI Optimized Image', use_column_width=True)
            else:
                st.info("The AI processed image preview will appear here after extraction.")

        st.divider()

        # Button to trigger extraction
        if st.button(f"1. Extract & Validate {selected_document_type.replace('_', ' ')} Data", type="primary"):
            
            st.session_state['extracted_data'] = None 
            st.session_state['accuracy_score'] = 1.0
            
            with st.spinner(f"Processing image and calling Gemini for {selected_document_type.replace('_', ' ')}..."):
                # 2. Run the image through the pipeline for enhancement and orientation
                # Pass document_type to conditionally run NRC specific enhancement
                enhanced_bytes = process_image(st.session_state['uploaded_file_bytes'], document_type=selected_document_type)
                st.session_state['enhanced_image_bytes'] = enhanced_bytes
                
                # 3. Perform the extraction
                extracted_data = extract_kyc_data(enhanced_bytes, selected_document_type)
            
            if extracted_data:
                st.session_state['extracted_data'] = extracted_data
                st.rerun() 
                
        # --- Display Results, Validation, and Correction ---
        if st.session_state['extracted_data'] is not None:
            st.header("Results and Validation")
            
            if selected_document_type == 'NRC':
                render_nrc_results(st.session_state['extracted_data'])
            elif selected_document_type == 'Driving_License':
                render_dl_results(st.session_state['extracted_data'])
            elif selected_document_type == 'Passport':
                render_passport_results(st.session_state['extracted_data'])

            # --- Download Logic ---
            st.subheader("💾 Download Final JSON")
            
            # Prepare data for download
            download_data = st.session_state['extracted_data'].copy()
            if 'validation_warnings' in download_data:
                download_data.pop('validation_warnings', None)
            
            if selected_document_type == 'NRC':
                download_data['OCR_Accuracy_Score'] = st.session_state.get('accuracy_score', 1.0)
            
            json_output = json.dumps(download_data, indent=2)

            st.download_button(
                label=f"⬇️ Download {selected_document_type.replace('_', ' ')} Data (JSON)",
                data=json_output,
                file_name=f"{selected_document_type.lower()}_data.json",
                mime="application/json",
            )
            
        st.markdown("---")

    else:
        st.info("Please select a document type, upload an image file or take a photo, and click '1. Extract & Validate Data' to begin.")


if __name__ == "__main__":
    main()