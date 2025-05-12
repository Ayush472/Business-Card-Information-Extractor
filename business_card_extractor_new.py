"""
Enhanced Business Card Information Extractor

This script extracts contact information from scanned business cards in a PDF file.
It uses improved OCR to read text from the images and enhanced pattern matching and NER
to identify and categorize information.

Usage:
    python enhanced_business_card_extractor.py input.pdf output.csv

Requirements:
    - Python 3.7+
    - pdf2image
    - pytesseract
    - opencv-python
    - spacy
    - pandas
    - tqdm
    - re
    - numpy
"""

import os
import re
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
import spacy
from concurrent.futures import ThreadPoolExecutor

# Load NER model for entity recognition
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Pre-compile regex patterns for better performance
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'(?:(?:\+|00)?\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{3,4}(?:[-.\s]?\d{1,4})?')
WEBSITE_PATTERN = re.compile(r'(?:https?:\/\/)?(?:www\.)?([A-Za-z0-9][-A-Za-z0-9]*[A-Za-z0-9]\.)+[A-Za-z]{2,}(?:\/[A-Za-z0-9\-._~:/?#[\]@!$&\'()*+,;=]*)?')
POST_CODE_PATTERN = re.compile(r'\b\d{5,6}(?:[-\s]\d{4})?\b')

# Common title keywords for detection
TITLE_KEYWORDS = [
    'ceo', 'cto', 'cfo', 'coo', 'president', 'vp', 'vice president', 
    'director', 'manager', 'head', 'chief', 'officer', 'executive',
    'associate', 'assistant', 'sr', 'senior', 'junior', 'jr', 'lead',
    'specialist', 'coordinator', 'analyst', 'consultant', 'advisor',
    'engineer', 'developer', 'architect', 'designer', 'technician',
    'supervisor', 'administrator', 'representative', 'sales', 'marketing',
    'purchase', 'procurement', 'hr', 'finance', 'operations', 'research',
    'development', 'project', 'product', 'general', 'regional', 'global'
]

# Common designator words for companies
COMPANY_DESIGNATORS = [
    'ltd', 'limited', 'inc', 'incorporated', 'llc', 'llp', 'lp', 'corp',
    'corporation', 'co', 'company', 'gmbh', 'ag', 'nv', 'sa', 'pty',
    'proprietary', 'pvt', 'private', 'plc', 'public', 'pte', 'bv', 'group',
    'holdings', 'industries', 'international', 'systems', 'solutions',
    'technologies', 'enterprises', 'associates', 'consultants', 'services',
    'partners', 'ventures', 'investments', 'global', 'worldwide', 'national',
    'chemicals', 'pharmaceuticals', 'engineering', 'manufacturing', 'trading'
]

def enhance_image(image):
    """Apply advanced image preprocessing for better OCR results"""
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Use a larger kernel for closing to connect nearby text
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Thin out the text for better OCR
    kernel = np.ones((1, 1), np.uint8)
    thinned = cv2.erode(closed, kernel, iterations=1)
    
    # Apply dilation to make text clearer
    kernel = np.ones((1, 1), np.uint8)
    final = cv2.dilate(thinned, kernel, iterations=1)
    
    return final

def extract_text_from_image(image):
    """Extract text from an image using improved OCR settings"""
    # Apply enhanced preprocessing
    processed_img = enhance_image(image)
    
    # Prepare for OCR - resize to improve detection
    scale_factor = 2.0
    resized = cv2.resize(processed_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Apply multiple OCR passes with different settings for better results
    # First pass - standard mode
    text1 = pytesseract.image_to_string(
        resized,
        config='--psm 6 --oem 3 -l eng+fra+deu --dpi 300',
        output_type=pytesseract.Output.STRING
    )
    
    # Second pass - with different page segmentation mode
    text2 = pytesseract.image_to_string(
        resized,
        config='--psm 4 --oem 3 -l eng+fra+deu --dpi 300',
        output_type=pytesseract.Output.STRING
    )
    
    # Third pass - with 11 (sparse text) mode
    text3 = pytesseract.image_to_string(
        resized,
        config='--psm 11 --oem 3 -l eng+fra+deu --dpi 300',
        output_type=pytesseract.Output.STRING
    )
    
    # Fourth pass - with 3 (fully automatic page segmentation) mode
    text4 = pytesseract.image_to_string(
        resized,
        config='--psm 3 --oem 3 -l eng+fra+deu --dpi 300',
        output_type=pytesseract.Output.STRING
    )
    
    # Combine results - join all texts and then clean
    combined_text = f"{text1}\n{text2}\n{text3}\n{text4}"
    
    # Clean up the extracted text
    cleaned_text = combined_text.replace('\n\n', '\n')
    cleaned_text = re.sub(r'[^\S\n]+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # Replace multiple newlines with single newline
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def clean_text(text):
    """Clean and normalize text"""
    # Replace common OCR errors
    text = text.replace('|', 'I').replace('l', 'l')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c == '\n')
    
    return text.strip()

def extract_emails(text):
    """Extract all email addresses from text"""
    return EMAIL_PATTERN.findall(text)

def extract_phones(text):
    """Extract and clean phone numbers"""
    # First, standardize common phone number separators
    text = re.sub(r'[\.\s\-()]+', ' ', text)
    
    # Find all potential phone numbers
    phones = PHONE_PATTERN.findall(text)
    
    # Clean and format phone numbers
    cleaned_phones = []
    for phone in phones:
        # Remove any non-digit characters except + and -
        cleaned = re.sub(r'[^\d+\-]', '', phone)
        
        # Validate the phone number (must have at least 7 digits)
        if sum(c.isdigit() for c in cleaned) >= 7:
            # Format phone numbers with proper spacing
            if cleaned.startswith('+'):
                # International format
                if len(cleaned) > 12:
                    cleaned = cleaned[:3] + ' ' + cleaned[3:6] + ' ' + cleaned[6:10] + ' ' + cleaned[10:]
                else:
                    cleaned = cleaned[:3] + ' ' + cleaned[3:]
            else:
                # Add spacing for readability
                if len(cleaned) >= 10:
                    cleaned = cleaned[:4] + ' ' + cleaned[4:7] + ' ' + cleaned[7:]
                    
            # Remove any duplicate spaces
            cleaned = ' '.join(cleaned.split())
            
            # Check for duplicates before adding
            if cleaned not in cleaned_phones:
                cleaned_phones.append(cleaned)
    
    return cleaned_phones

def extract_websites(text):
    """Extract website URLs from text"""
    websites = WEBSITE_PATTERN.findall(text)
    cleaned_websites = []
    
    for website in websites:
        # Clean and format website URL
        if website:
            # Ensure URL has proper prefix
            if not website.startswith(('http://', 'https://', 'www.')):
                website = 'www.' + website
            
            # Remove any non-standard characters
            website = re.sub(r'[^\w\-\.]', '', website)
            
            # Ensure website has a valid domain
            if '.' in website and len(website.split('.')[-1]) >= 2:
                cleaned_websites.append(website)
    
    return cleaned_websites

def extract_potential_names(text, doc):
    """Extract potential person names using various methods"""
    potential_names = []
    
    # Method 1: Use NER to identify PERSON entities
    person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    potential_names.extend(person_entities)
    
    # Method 2: Look for common name patterns
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Check for short lines that might be names (2-3 words, no digits)
        words = line.split()
        if 1 <= len(words) <= 3 and all(not any(c.isdigit() for c in word) for word in words):
            # Check for capitalization patterns (First Last)
            if all(word[0].isupper() if word else False for word in words):
                # Skip if contains title keywords
                if not any(title.lower() in line.lower() for title in TITLE_KEYWORDS):
                    potential_names.append(line)
    
    # Method 3: Look for lines with common name prefixes
    name_prefixes = ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'miss']
    for line in lines:
        line = line.strip()
        lower_line = line.lower()
        if any(lower_line.startswith(prefix) for prefix in name_prefixes):
            potential_names.append(line)
    
    # Clean the potential names
    cleaned_names = []
    for name in potential_names:
        # Remove any punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s]', '', name)
        cleaned = ' '.join(cleaned.split())
        
        # Skip if too short or contains digits
        if len(cleaned) >= 3 and not any(c.isdigit() for c in cleaned):
            cleaned_names.append(cleaned)
    
    return cleaned_names

def extract_potential_titles(text, doc):
    """Extract potential job titles using various methods"""
    potential_titles = []
    
    # Method 1: Look for lines with title keywords
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        lower_line = line.lower()
        
        # Check if line contains any title keywords
        if any(title.lower() in lower_line for title in TITLE_KEYWORDS):
            # Skip if it's likely part of a company name
            if not any(designator.lower() in lower_line for designator in COMPANY_DESIGNATORS):
                potential_titles.append(line)
    
    # Method 2: Use NER context - look for lines near PERSON entities
    person_indices = [i for i, line in enumerate(lines) if any(ent.text in line for ent in doc.ents if ent.label_ == "PERSON")]
    for idx in person_indices:
        # Check surrounding lines
        for i in range(max(0, idx-1), min(len(lines), idx+2)):
            line = lines[i].strip()
            if line and line not in potential_titles:
                lower_line = line.lower()
                if any(title.lower() in lower_line for title in TITLE_KEYWORDS):
                    potential_titles.append(line)
    
    # Clean the potential titles
    cleaned_titles = []
    for title in potential_titles:
        # Remove any punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s\-]', '', title)
        cleaned = ' '.join(cleaned.split())
        
        # Skip if too short or identical to company name
        if len(cleaned) >= 3:
            cleaned_titles.append(cleaned)
    
    return cleaned_titles

def extract_potential_companies(text, doc):
    """Extract potential company names using various methods"""
    potential_companies = []
    
    # Method 1: Use NER to identify ORG entities
    org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    potential_companies.extend(org_entities)
    
    # Method 2: Look for lines with company designators
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        lower_line = line.lower()
        
        # Check if line contains any company designators
        if any(designator.lower() in lower_line for designator in COMPANY_DESIGNATORS):
            potential_companies.append(line)
    
    # Method 3: Look for lines with all caps (common for company names)
    for line in lines:
        line = line.strip()
        if line.isupper() and len(line) > 3 and not any(c.isdigit() for c in line):
            potential_companies.append(line)
    
    # Clean the potential companies
    cleaned_companies = []
    for company in potential_companies:
        # Remove any punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s\-&.,]', '', company)
        cleaned = ' '.join(cleaned.split())
        
        # Skip if too short
        if len(cleaned) >= 3:
            cleaned_companies.append(cleaned)
    
    return cleaned_companies

def extract_potential_addresses(text, doc):
    """Extract potential addresses using various methods"""
    potential_addresses = []
    
    # Method 1: Use NER to identify GPE and LOC entities
    loc_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    
    # Method 2: Look for lines with location indicators
    address_indicators = ['road', 'street', 'avenue', 'lane', 'plot', 'sector', 'area', 'city', 
                         'state', 'country', 'district', 'floor', 'suite', 'unit', 'building',
                         'block', 'postal', 'zip', 'code', 'highway', 'junction']
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        lower_line = line.lower()
        
        # Check if line has address indicators or postal codes
        has_indicators = any(indicator in lower_line for indicator in address_indicators)
        has_postal_code = bool(POST_CODE_PATTERN.search(line))
        
        if has_indicators or has_postal_code:
            # Include surrounding lines for context
            address_block = []
            for j in range(max(0, i-1), min(len(lines), i+2)):
                if lines[j].strip():
                    address_block.append(lines[j].strip())
            
            potential_addresses.append(' '.join(address_block))
    
    # Method 3: Look for lines with GPE or LOC entities
    address_lines = []
    for entity in loc_entities:
        # Find all lines containing this entity
        for line in lines:
            if entity in line and line not in address_lines:
                address_lines.append(line)
    
    if address_lines:
        potential_addresses.append(' '.join(address_lines))
    
    # Clean the potential addresses
    cleaned_addresses = []
    for address in potential_addresses:
        # Remove any extra spaces
        cleaned = ' '.join(address.split())
        
        # Skip if too short
        if len(cleaned) >= 10:
            cleaned_addresses.append(cleaned)
    
    return cleaned_addresses

def extract_structured_data(text):
    """Extract structured information using improved pattern matching and NER"""
    # Initialize the output data structure
    data = {
        "name": None,
        "title": None,
        "company": None,
        "phone": [],
        "email": None,
        "website": None,
        "address": None,
        "other": []
    }
    
    # Clean the text
    text = clean_text(text)
    
    # Process with spaCy for NER
    doc = nlp(text)
    
    # Extract emails
    emails = extract_emails(text)
    if emails:
        data["email"] = emails[0]
    
    # Extract phone numbers
    data["phone"] = extract_phones(text)
    
    # Extract websites
    websites = extract_websites(text)
    if websites:
        data["website"] = websites[0]
    
    # Extract potential names
    potential_names = extract_potential_names(text, doc)
    if potential_names:
        data["name"] = potential_names[0]
    
    # Extract potential job titles
    potential_titles = extract_potential_titles(text, doc)
    if potential_titles:
        data["title"] = potential_titles[0]
    
    # Extract potential company names
    potential_companies = extract_potential_companies(text, doc)
    if potential_companies:
        data["company"] = potential_companies[0]
    
    # Extract potential addresses
    potential_addresses = extract_potential_addresses(text, doc)
    if potential_addresses:
        data["address"] = potential_addresses[0]
    
    # Collect other information
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 3:  # Skip very short lines
            # Check if this line is already captured in any field
            already_captured = False
            for key, value in data.items():
                if value:
                    if isinstance(value, list):
                        if any(line in item for item in value):
                            already_captured = True
                            break
                    elif line in value:
                        already_captured = True
                        break
            
            if not already_captured:
                # Skip if it contains known patterns like email or phone
                if not EMAIL_PATTERN.search(line) and not PHONE_PATTERN.search(line):
                    # Clean and add to other information
                    cleaned = ' '.join(line.split())
                    if cleaned and cleaned not in data["other"]:
                        data["other"].append(cleaned)
    
    return data

def process_card(image, index):
    """Process a single business card image with enhanced processing"""
    try:
        # Extract text using improved OCR
        text = extract_text_from_image(image)
        
        # Extract structured data with improved extraction logic
        data = extract_structured_data(text)
        
        # Add raw text and card index
        data["raw_text"] = text
        data["card_index"] = index
        
        return data
    except Exception as e:
        print(f"Error processing card {index}: {e}")
        return {
            "card_index": index,
            "error": str(e),
            "raw_text": "",
            "name": None,
            "title": None,
            "company": None,
            "phone": [],
            "email": None,
            "website": None,
            "address": None,
            "other": []
        }

def process_pdf(pdf_path, output_path=None, max_workers=None):
    """Process all business cards in a PDF file"""
    print(f"Converting PDF to images: {pdf_path}")
    
    # Convert PDF to images with higher DPI
    try:
        images = convert_from_path(
            pdf_path,
            dpi=600,  # Higher DPI for better OCR
            thread_count=os.cpu_count(),
            grayscale=False  # Keep color for better processing
        )
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("Attempting with lower DPI...")
        images = convert_from_path(
            pdf_path,
            dpi=300,  # Lower DPI as fallback
            thread_count=os.cpu_count(),
            grayscale=False
        )
    
    print(f"Total images extracted: {len(images)}")
    
    # Process each card
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_card, image, i): i for i, image in enumerate(images)}
        
        for future in tqdm(futures, desc="Processing cards"):
            results.append(future.result())
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "Card Index": r["card_index"],
            "Name": r["name"],
            "Title": r["title"],
            "Company": r["company"],
            "Phone": ", ".join(r["phone"]) if r["phone"] else None,
            "Email": r["email"],
            "Website": r["website"],
            "Address": r["address"],
            "Other Information": ", ".join(r["other"]) if r["other"] else None,
            "Raw Text": r["raw_text"]
        }
        for r in results
    ])
    
    # Sort by card index
    df = df.sort_values("Card Index")
    
    # Output to CSV if specified
    if output_path:
        print(f"Saving results to: {output_path}")
        df.to_csv(output_path, index=False)
        
        # Also save as JSON for more complete data
        json_path = output_path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved complete JSON data to: {json_path}")
    
    return df, results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract contact information from business cards")
    parser.add_argument("pdf_path", help="Path to the PDF file containing business cards")
    parser.add_argument("output_path", help="Path to save the output CSV file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Check if pytesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("Error: Tesseract OCR is not installed or not in PATH.")
        print("Please install Tesseract OCR and make sure it's in your PATH.")
        print("Installation guide: https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)
    
    # Process the PDF
    df, _ = process_pdf(args.pdf_path, args.output_path, args.workers)
    
    # Print summary
    print("\nExtraction Summary:")
    print(f"Total cards processed: {len(df)}")
    print(f"Cards with names: {df['Name'].notnull().sum()} ({df['Name'].notnull().sum()/len(df)*100:.1f}%)")
    print(f"Cards with emails: {df['Email'].notnull().sum()} ({df['Email'].notnull().sum()/len(df)*100:.1f}%)")
    print(f"Cards with phone numbers: {df['Phone'].notnull().sum()} ({df['Phone'].notnull().sum()/len(df)*100:.1f}%)")
    print(f"Cards with company names: {df['Company'].notnull().sum()} ({df['Company'].notnull().sum()/len(df)*100:.1f}%)")
    
    print("\nSample of extracted data:")
    print(df.head(3))

if __name__ == "__main__":
    main()