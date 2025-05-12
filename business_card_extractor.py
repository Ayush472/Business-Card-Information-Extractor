"""
Business Card Information Extractor

This script extracts contact information from scanned business cards in a PDF file.
It uses OCR to read text from the images and NER to identify and categorize information.

Usage:
    python business_card_extractor.py input.pdf output.csv

Requirements:
    - Python 3.7+
    - pdf2image
    - pytesseract
    - opencv-python
    - spacy
    - pandas
    - tqdm
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

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert to numpy array
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation to make text more prominent
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    
    return dilated

def extract_text_from_image(image):
    """Extract text from an image using OCR"""
    processed_img = preprocess_image(image)
    
    # Apply OCR with optimized settings
    text = pytesseract.image_to_string(
        processed_img,
        config='--psm 6 --oem 3 -l eng+fra+deu --dpi 300',
        output_type=pytesseract.Output.STRING
    )
    
    # Clean up the extracted text
    text = text.replace('\n\n', '\n')  # Remove double newlines
    text = re.sub(r'[^\S\n]+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def extract_structured_data(text):
    """Extract structured information using NER and regex patterns"""
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
    
    # Clean text
    text = text.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Extract email using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        data["email"] = emails[0]
    
    # Extract phone numbers using regex - improved pattern
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{4,}'
    phones = re.findall(phone_pattern, text)
    if phones:
        # Clean and format phone numbers
        cleaned_phones = []
        for phone in phones:
            # Remove any non-digit characters except + and -
            cleaned = re.sub(r'[^\d+-]', '', phone)
            # Ensure proper formatting
            if cleaned.startswith('+91'):
                cleaned = cleaned.replace('+91', '+91 ')
            # Remove any duplicate spaces
            cleaned = ' '.join(cleaned.split())
            if cleaned not in cleaned_phones:  # Avoid duplicates
                cleaned_phones.append(cleaned)
        data["phone"] = cleaned_phones
    
    # Extract website using regex - improved pattern
    website_pattern = r'(?:https?:\/\/)?(?:www\.)?([A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:\/[A-Za-z0-9-._~:/?#[\]@!$&\'()*+,;=]*)?'
    websites = re.findall(website_pattern, text)
    if websites:
        # Clean and format website
        website = websites[0]
        if not website.startswith('http'):
            website = 'www.' + website
        # Only set website if it's a valid domain
        if '.' in website and len(website.split('.')[-1]) >= 2:
            data["website"] = website
    
    # Use NER for person name, organization, and location
    doc = nlp(text)
    
    # Extract person names - improved logic
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if person_names:
        # Clean and format name
        name = person_names[0]
        # Remove any special characters and extra spaces
        name = re.sub(r'[^\w\s.]', '', name)
        name = ' '.join(name.split())
        data["name"] = name
    
    # Extract organizations - improved logic
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if orgs:
        # Clean and format company name
        company = orgs[0]
        # Remove any special characters and extra spaces
        company = re.sub(r'[^\w\s.,&]', '', company)
        company = ' '.join(company.split())
        data["company"] = company
    
    # Extract locations (potential address) - improved logic
    locs = [ent.text for ent in doc.ents if ent.label_ == "GPE" or ent.label_ == "LOC"]
    if locs:
        # Clean and format address
        address = ", ".join(locs)
        # Remove any special characters and extra spaces
        address = re.sub(r'[^\w\s.,-]', '', address)
        address = ' '.join(address.split())
        data["address"] = address
    
    # If name is not found by NER, use heuristics
    if not data["name"]:
        # Look for lines that might contain a name
        for line in lines:
            # Check if line looks like a name (contains title or is short)
            if len(line.split()) <= 4 or any(title in line.lower() for title in 
                ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam']):
                # Clean and format name
                name = re.sub(r'[^\w\s.]', '', line)
                name = ' '.join(name.split())
                if name and not any(char.isdigit() for char in name):  # Avoid lines with numbers
                    data["name"] = name
                    break
    
    # If title is not found, use heuristics
    if not data["title"]:
        # Look for lines that might contain a title
        for line in lines:
            # Skip if line looks like a company name or contains numbers
            if not any(company_indicator in line.lower() for company_indicator in 
                      ['ltd', 'inc', 'corp', 'llc', 'limited', 'company', 'co.', 'corporation']) and \
               not any(char.isdigit() for char in line):
                # Check for common title indicators
                if any(title in line.lower() for title in 
                      ['manager', 'director', 'head', 'president', 'vice', 'chief', 'officer', 'engineer']):
                    # Clean and format title
                    title = re.sub(r'[^\w\s.,-]', '', line)
                    title = ' '.join(title.split())
                    data["title"] = title
                    break
    
    # If company is not found, use heuristics
    if not data["company"]:
        for line in lines:
            if any(company_indicator in line.lower() for company_indicator in 
                  ['ltd', 'inc', 'corp', 'llc', 'limited', 'company', 'co.', 'corporation']):
                # Clean and format company name
                company = re.sub(r'[^\w\s.,&]', '', line)
                company = ' '.join(company.split())
                data["company"] = company
                break
    
    # If address is not found, use heuristics
    if not data["address"]:
        address_lines = []
        for line in lines:
            # Look for lines that might contain an address
            if any(addr_indicator in line.lower() for addr_indicator in 
                  ['road', 'street', 'avenue', 'lane', 'plot', 'sector', 'area', 'city', 'state']):
                # Clean and format address line
                addr_line = re.sub(r'[^\w\s.,-]', '', line)
                addr_line = ' '.join(addr_line.split())
                if addr_line:
                    address_lines.append(addr_line)
        if address_lines:
            data["address"] = ', '.join(address_lines)
    
    # Collect other information not categorized
    for line in lines:
        categorized = False
        for value in data.values():
            if isinstance(value, list):
                if line in value:
                    categorized = True
                    break
            elif value and line in value:
                categorized = True
                break
        
        if not categorized and line and len(line) > 3:  # Skip very short lines
            # Clean and format other information
            other = re.sub(r'[^\w\s.,-]', '', line)
            other = ' '.join(other.split())
            if other and other not in data["other"]:  # Avoid duplicates
                data["other"].append(other)
    return data

def process_card(image, index):
    """Process a single business card image"""
    try:
        # Extract text
        text = extract_text_from_image(image)
        
        # Extract structured data
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
    
    # Convert PDF to images
    images = convert_from_path(
        pdf_path,
        dpi=300,  # Higher DPI for better OCR
        thread_count=os.cpu_count()
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
    print(f"Cards with names: {df['Name'].notnull().sum()}")
    print(f"Cards with emails: {df['Email'].notnull().sum()}")
    print(f"Cards with phone numbers: {df['Phone'].notnull().sum()}")
    print(f"Cards with company names: {df['Company'].notnull().sum()}")
    
    print("\nSample of extracted data:")
    print(df.head(3))

if __name__ == "__main__":
    main()