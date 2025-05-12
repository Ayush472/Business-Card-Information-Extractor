"""
Business Card Extractor - Advanced Self-Training Model

This script extends the base extractor with a feedback loop and self-training capabilities.
It allows users to correct extraction errors, which helps train the system to improve
accuracy for similar cards in the future.

Usage:
    python advanced_extractor.py --train --pdf input.pdf --output output.csv
"""

import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import the base extractor functionality
from business_card_extractor import process_pdf, extract_structured_data

class AdaptiveCardExtractor:
    """A self-training business card information extractor"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models for different fields
        self.field_models = {
            "name": self._load_model("name_model"),
            "title": self._load_model("title_model"),
            "company": self._load_model("company_model"),
            "email": self._load_model("email_model"),
            "phone": self._load_model("phone_model"),
            "website": self._load_model("website_model"),
            "address": self._load_model("address_model")
        }
        
        # Initialize vectorizers
        self.vectorizers = {
            field: self._load_vectorizer(f"{field}_vectorizer")
            for field in self.field_models.keys()
        }
        
        # Training data
        self.training_data = self._load_training_data()
    
    def _load_model(self, model_name):
        """Load a trained model or create a new one if it doesn't exist"""
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
        
        # Return a new model
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _load_vectorizer(self, vectorizer_name):
        """Load a trained vectorizer or create a new one"""
        vec_path = os.path.join(self.model_dir, f"{vectorizer_name}.pkl")
        if os.path.exists(vec_path):
            try:
                with open(vec_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading vectorizer {vectorizer_name}: {e}")
        
        # Return a new vectorizer
        return TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1, 2))
    
    def _load_training_data(self):
        """Load existing training data or initialize empty"""
        data_path = os.path.join(self.model_dir, "training_data.json")
        if os.path.exists(data_path):
            try:
                with open(data_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading training data: {e}")
        
        # Initialize empty training data
        return {
            field: {"texts": [], "labels": []} 
            for field in self.field_models.keys()
        }
    
    def _save_model(self, model, model_name):
        """Save a trained model"""
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
    def _save_vectorizer(self, vectorizer, vectorizer_name):
        """Save a trained vectorizer"""
        vec_path = os.path.join(self.model_dir, f"{vectorizer_name}.pkl")
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)
    
    def _save_training_data(self):
        """Save training data"""
        data_path = os.path.join(self.model_dir, "training_data.json")
        with open(data_path, "w") as f:
            json.dump(self.training_data, f, indent=2)
    
    def add_training_example(self, text, field_values):
        """Add a new training example"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for field, value in field_values.items():
            if field not in self.training_data:
                continue
                
            # Add positive examples
            if value:
                for line in lines:
                    # Check if this line contains the target value
                    if value.lower() in line.lower():
                        self.training_data[field]["texts"].append(line)
                        self.training_data[field]["labels"].append(1)
                    else:
                        self.training_data[field]["texts"].append(line)
                        self.training_data[field]["labels"].append(0)
        
        # Save updated training data
        self._save_training_data()
    
    def train_models(self):
        """Train all models with current training data"""
        for field, model in self.field_models.items():
            if field not in self.training_data:
                continue
                
            texts = self.training_data[field]["texts"]
            labels = self.training_data[field]["labels"]
            
            if not texts or len(set(labels)) < 2:
                print(f"Not enough diverse training data for {field}")
                continue
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42)
            
            # Create or fit vectorizer
            vectorizer = self.vectorizers[field]
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_vec)
            print(f"\nModel performance for {field}:")
            print(classification_report(y_test, y_pred))
            
            # Save trained model and vectorizer
            self._save_model(model, f"{field}_model")
            self._save_vectorizer(vectorizer, f"{field}_vectorizer")
            
            # Update instance
            self.field_models[field] = model
            self.vectorizers[field] = vectorizer
    
    def enhance_extraction(self, extracted_data, raw_text):
        """Enhance extraction using trained models"""
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        for field, model in self.field_models.items():
            # Skip if field is already extracted with high confidence
            if field in extracted_data and extracted_data[field]:
                continue
                
            try:
                # Vectorize lines
                vectorizer = self.vectorizers[field]
                if not hasattr(vectorizer, 'vocabulary_'):
                    # Vectorizer not trained yet
                    continue
                    
                X = vectorizer.transform(lines)
                
                # Predict
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]  # Probability of positive class
                
                # Get the highest probability prediction
                if any(predictions):
                    best_idx = np.argmax(probabilities)
                    if probabilities[best_idx] > 0.7:  # Confidence threshold
                        extracted_data[field] = lines[best_idx]
            except Exception as e:
                print(f"Error enhancing extraction for {field}: {e}")
        
        return extracted_data

def process_with_feedback(pdf_path, output_path, extractor=None, training_mode=False):
    """Process PDF with feedback loop for continuous improvement"""
    # Create extractor if not provided
    if extractor is None:
        extractor = AdaptiveCardExtractor()
    
    # Process PDF using base functionality
    df, raw_results = process_pdf(pdf_path, output_path)
    
    # If in training mode, ask for corrections
    if training_mode:
        print("\n--- Training Mode Activated ---")
        print("You will now be asked to verify and correct the extracted information.")
        print("This will help train the system for better future extractions.")
        
        for i, result in enumerate(raw_results):
            print(f"\n--- Card {i+1}/{len(raw_results)} ---")
            print(f"Raw text:\n{result['raw_text']}\n")
            
            corrected_data = {}
            
            # Ask for corrections for each field
            for field in ["name", "title", "company", "email", "website", "address"]:
                current_value = result[field]
                if isinstance(current_value, list) and current_value:
                    current_value = current_value[0]
                
                print(f"Current {field}: {current_value}")
                correction = input(f"Correct {field} (press Enter to keep current): ")
                
                if correction:
                    corrected_data[field] = correction
                else:
                    corrected_data[field] = current_value
            
            # Add training example
            extractor.add_training_example(result["raw_text"], corrected_data)
            
            # Check if user wants to continue
            if i < len(raw_results) - 1:
                cont = input("\nContinue to next card? (y/n): ")
                if cont.lower() != 'y':
                    break
        
        # Train models with new data
        print("\nTraining models with new data...")
        extractor.train_models()
        print("Training complete!")
    
    return df, raw_results, extractor

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced business card information extractor")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file containing business cards")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument("--train", action="store_true", help="Activate training mode")
    
    args = parser.parse_args()
    
    # Process with feedback
    extractor = AdaptiveCardExtractor()
    df, results, trained_extractor = process_with_feedback(
        args.pdf, args.output, extractor, args.train)
    
    # Print summary
    print("\nExtraction Summary:")
    print(f"Total cards processed: {len(df)}")
    print(f"Cards with names: {df['Name'].notnull().sum()}")
    print(f"Cards with emails: {df['Email'].notnull().sum()}")
    print(f"Cards with phone numbers: {df['Phone'].notnull().sum()}")
    print(f"Cards with company names: {df['Company'].notnull().sum()}")
    
    print("\nDone! Check the output files for results.")

if __name__ == "__main__":
    main()