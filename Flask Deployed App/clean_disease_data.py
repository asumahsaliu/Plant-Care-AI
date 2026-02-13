"""
CSV Dataset Cleaning Script for Plant Disease Information
===========================================================

This script cleans and optimizes the disease_info.csv dataset for ML usage:
1. Splits plant and disease names
2. Fixes broken/truncated text
3. Removes special characters
4. Standardizes formatting
5. Handles missing data
6. Ensures CSV integrity
"""

import pandas as pd
import re
import csv

def clean_text(text):
    """Remove special characters and fix encoding issues"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove special characters like �
    text = text.replace('�', "'")
    text = text.replace('�', '"')
    text = text.replace('�', '"')
    text = text.replace('�', '-')
    text = text.replace('�', '...')
    
    # Remove other problematic characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def split_disease_name(disease_name):
    """Split 'Plant : Disease' into separate columns"""
    if pd.isna(disease_name) or ':' not in disease_name:
        return 'Unknown', disease_name
    
    parts = disease_name.split(':', 1)
    plant = parts[0].strip()
    disease = parts[1].strip() if len(parts) > 1 else 'Unknown'
    
    return plant, disease

def standardize_steps(text):
    """Standardize formatting of possible steps"""
    if pd.isna(text) or text == '':
        return ''
    
    text = clean_text(text)
    
    # Split by periods or newlines to identify steps
    steps = re.split(r'\.(?:\s+|\n+)', text)
    
    # Clean each step
    cleaned_steps = []
    for step in steps:
        step = step.strip()
        if step and len(step) > 10:  # Filter out very short fragments
            # Remove leading numbers or bullets
            step = re.sub(r'^[\d\-\*\•]+\.?\s*', '', step)
            if not step.endswith('.'):
                step += '.'
            cleaned_steps.append(step)
    
    # Join with newlines for readability
    return ' '.join(cleaned_steps)

def is_placeholder_image(url):
    """Check if image URL is a placeholder"""
    if pd.isna(url) or url == '':
        return True
    
    placeholder_indicators = [
        'no leaf',
        'placeholder',
        'onlinewebfonts',
        'img_332817',
        'default',
        'missing'
    ]
    
    url_lower = url.lower()
    return any(indicator in url_lower for indicator in placeholder_indicators)

def check_completeness(text, min_length=50):
    """Check if text appears complete (not truncated)"""
    if pd.isna(text) or text == '':
        return False, 'MISSING'
    
    text = str(text).strip()
    
    # Check minimum length
    if len(text) < min_length:
        return False, 'TOO_SHORT'
    
    # Check if ends abruptly (no punctuation)
    if not text[-1] in '.!?':
        return False, 'TRUNCATED'
    
    return True, 'COMPLETE'

def clean_dataset(input_file, output_file):
    """Main function to clean the dataset"""
    
    print("Loading dataset...")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            print(f"  Trying encoding: {encoding}")
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"  ✓ Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise Exception("Could not read file with any standard encoding")
    
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Create new columns
    print("\n1. Splitting plant and disease names...")
    df[['Plant', 'Disease']] = df['disease_name'].apply(
        lambda x: pd.Series(split_disease_name(x))
    )
    
    # Clean text fields
    print("2. Cleaning text and removing special characters...")
    df['Description'] = df['description'].apply(clean_text)
    df['Possible_Steps'] = df['Possible Steps'].apply(clean_text)
    df['Possible_Steps'] = df['Possible_Steps'].apply(standardize_steps)
    
    # Clean image URLs
    print("3. Processing image URLs...")
    df['Image_URL'] = df['image_url'].apply(clean_text)
    df['Has_Valid_Image'] = ~df['Image_URL'].apply(is_placeholder_image)
    
    # Check completeness
    print("4. Checking data completeness...")
    df[['Description_Complete', 'Description_Status']] = df['Description'].apply(
        lambda x: pd.Series(check_completeness(x, min_length=50))
    )
    df[['Steps_Complete', 'Steps_Status']] = df['Possible_Steps'].apply(
        lambda x: pd.Series(check_completeness(x, min_length=30))
    )
    
    # Create final cleaned dataset
    print("5. Creating final dataset...")
    cleaned_df = pd.DataFrame({
        'Index': df['index'],
        'Plant': df['Plant'],
        'Disease': df['Disease'],
        'Description': df['Description'],
        'Possible_Steps': df['Possible_Steps'],
        'Image_URL': df['Image_URL'],
        'Has_Valid_Image': df['Has_Valid_Image'],
        'Description_Status': df['Description_Status'],
        'Steps_Status': df['Steps_Status']
    })
    
    # Save cleaned dataset
    print(f"\n6. Saving cleaned dataset to {output_file}...")
    cleaned_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    # Generate report
    print("\n" + "="*70)
    print("CLEANING REPORT")
    print("="*70)
    
    print(f"\nTotal entries: {len(cleaned_df)}")
    print(f"\nUnique plants: {cleaned_df['Plant'].nunique()}")
    print(f"Plant distribution:")
    print(cleaned_df['Plant'].value_counts().to_string())
    
    print(f"\n\nData Quality:")
    print(f"  - Valid images: {cleaned_df['Has_Valid_Image'].sum()} / {len(cleaned_df)}")
    print(f"  - Complete descriptions: {(cleaned_df['Description_Status'] == 'COMPLETE').sum()} / {len(cleaned_df)}")
    print(f"  - Complete steps: {(cleaned_df['Steps_Status'] == 'COMPLETE').sum()} / {len(cleaned_df)}")
    
    print(f"\n\nIssues found:")
    print(f"  - Missing descriptions: {(cleaned_df['Description_Status'] == 'MISSING').sum()}")
    print(f"  - Truncated descriptions: {(cleaned_df['Description_Status'] == 'TRUNCATED').sum()}")
    print(f"  - Missing steps: {(cleaned_df['Steps_Status'] == 'MISSING').sum()}")
    print(f"  - Truncated steps: {(cleaned_df['Steps_Status'] == 'TRUNCATED').sum()}")
    print(f"  - Placeholder images: {(~cleaned_df['Has_Valid_Image']).sum()}")
    
    # Show problematic entries
    print(f"\n\nProblematic entries:")
    problematic = cleaned_df[
        (~cleaned_df['Has_Valid_Image']) | 
        (cleaned_df['Description_Status'] != 'COMPLETE') |
        (cleaned_df['Steps_Status'] != 'COMPLETE')
    ]
    
    if len(problematic) > 0:
        print(problematic[['Index', 'Plant', 'Disease', 'Description_Status', 'Steps_Status', 'Has_Valid_Image']].to_string())
    else:
        print("  None found!")
    
    print("\n" + "="*70)
    print("CLEANING COMPLETE!")
    print("="*70)
    
    return cleaned_df

if __name__ == '__main__':
    input_file = 'disease_info.csv'
    output_file = 'disease_info_cleaned.csv'
    
    try:
        cleaned_df = clean_dataset(input_file, output_file)
        print(f"\n✓ Cleaned dataset saved to: {output_file}")
        print(f"✓ Ready for ML usage!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
