# Business Card Information Extractor - Setup Guide

This guide will help you set up and run the Business Card Information Extractor on your sample PDF of business cards.

## Prerequisites

Before running the script, you need to install the following software:

1. **Python 3.7 or higher**
   - Download from [python.org](https://www.python.org/downloads/)

2. **Tesseract OCR**
   - This is the OCR engine that powers text extraction
   
   **For Windows:**
   - Download the installer from [GitHub UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Run the installer and note the installation path
   - Add the Tesseract installation directory to your system PATH

   **For macOS:**
   ```
   brew install tesseract
   ```

   **For Linux:**
   ```
   sudo apt-get install tesseract-ocr
   ```

3. **Poppler** (required for PDF processing)
   
   **For Windows:**
   - Download the binary from [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Extract the contents and add the `bin` directory to your system PATH

   **For macOS:**
   ```
   brew install poppler
   ```

   **For Linux:**
   ```
   sudo apt-get install poppler-utils
   ```

## Installation Steps

1. **Create a new directory for the project**
   ```
   mkdir business-card-extractor
   cd business-card-extractor
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **For Windows:**
   ```
   venv\Scripts\activate
   ```

   **For macOS/Linux:**
   ```
   source venv/bin/activate
   ```

4. **Install required Python packages**
   ```
   pip install pdf2image pytesseract opencv-python spacy pandas tqdm
   ```

5. **Download SpaCy language model**
   ```
   python -m spacy download en_core_web_lg
   ```

6. **Save the script**
   - Save the provided Python script as `business_card_extractor.py` in your project directory

## Running the Extractor

1. **Place your sample PDF in the project directory**

2. **Run the script**
   ```
   python business_card_extractor.py Scan.pdf output.csv
   ```
   Replace `sample.pdf` with the path to your PDF file containing the business cards.

3. **Check the output**
   - The script will generate two files:
     - `output.csv`: Contains the structured data in CSV format
     - `output.json`: Contains more detailed information in JSON format

## Understanding the Output

The CSV file will contain the following columns:
- Card Index: The position of the card in the PDF
- Name: Extracted person name
- Title: Job title or position
- Company: Company or organization name
- Phone: Extracted phone numbers
- Email: Email address
- Website: Company website
- Address: Physical address information
- Other Information: Any other text that couldn't be categorized
- Raw Text: All text extracted from the card

## Improving Accuracy

If you find the extraction accuracy needs improvement:

1. **Improve scan quality**
   - Ensure your scans are high resolution (300 DPI or higher)
   - Make sure business cards are well-aligned and have good contrast

2. **Modify preprocessing parameters**
   - Adjust the preprocessing parameters in the `preprocess_image()` function

3. **Customize extraction logic**
   - For specific card formats or information requirements, you may want to customize the `extract_structured_data()` function

## Scaling to Your Full Dataset

Once you're satisfied with the extraction quality on your sample set:

1. Run the script on your complete PDF file containing all 2,000+ cards
2. Be patient - processing a large number of cards will take time
3. Consider running the process overnight for very large datasets

## Troubleshooting

If you encounter errors:

1. **Tesseract not found**
   - Ensure Tesseract is installed and in your system PATH
   - You can specify the Tesseract path directly in the code if needed

2. **Memory errors**
   - For very large PDFs, you might need to process in batches
   - Consider adding pagination logic to process a few pages at a time

3. **Extraction accuracy issues**
   - Try adjusting the preprocessing parameters
   - Consider training a custom NER model if you have specific naming patterns

## Getting the Data into a Database

Once you have your CSV file:

1. **Import to spreadsheet**
   - You can open the CSV in Excel or Google Sheets for manual review

2. **Import to database**
   - Use the database of your choice (SQL, MongoDB, etc.)
   - Most database systems have CSV import functionality

3. **Build a simple interface**
   - Consider building a simple web interface to search and manage your contacts