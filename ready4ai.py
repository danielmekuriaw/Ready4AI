#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
import json
import sys
import requests
from datetime import datetime, timedelta

# Valid US States (for simple checking)
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
    'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV',
    'NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
    'TX','UT','VT','VA','WA','WV','WI','WY'
}

def pretty_print_json(data, indent=0):
    """Recursively pretty print JSON-like data without braces or quotes."""
    prefix = " " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:", end=" ")
            
            if isinstance(value, (dict, list)):
                print()
                pretty_print_json(value, indent + 4)
                print()
            
            else:
                print(value)
    
    elif isinstance(data, list):
        for item in data:
            pretty_print_json(item, indent)

# Data Profiling
def profile_data(file_path, consider_cols=None, shield_cols=None):
    """
    Loads the CSV and profiles the data.
    - If consider_cols is provided, only these columns are profiled.
    - If shield_cols is provided, these columns are removed from profiling and flagged.
    Returns a profile dictionary and saves it to 'profile_results.json'.
    """
    try:
        df = pd.read_csv(file_path)
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    original_columns = list(df.columns)

    # Shielding: removing specified columns from the dataset
    shielded = []
    
    if shield_cols:
        
        shield_cols = [col.strip() for col in shield_cols.split(",")]
        
        for col in shield_cols:
            if col not in df.columns:
                print(f"Warning: Shield column '{col}' not found in dataset.")
            
            else:
                shielded.append(col)
                df.drop(columns=[col], inplace=True)

    # Considering: if provided, only profile these columns
    if consider_cols:
        consider_cols = [col.strip() for col in consider_cols.split(",")]
        missing = [col for col in consider_cols if col not in df.columns]
        
        if missing:
            print(f"Warning: Consider columns {missing} not found in dataset. Ignoring them.")
        
        df = df[[col for col in consider_cols if col in df.columns]]
        
    intended_use = input("Enter the intended use of the dataset (e.g., customer segmentation, fraud detection): ")
    ai_system = input("Enter the target AI system (e.g., recommendation engine, NLP chatbot, Vision Model, World Model): ")
    training_testing_ft = input("Will it be used for training, testing, fine-tuning or neither?: ")

    # Dictionary to store the results
    profile_results = {}

    # Basic dimensions
    profile_results["shape"] = df.shape

    # Missing values per column
    profile_results["missing_values"] = df.isnull().sum().to_dict()

    # Duplicate rows count
    profile_results["duplicate_rows"] = int(df.duplicated().sum())

    # Data types and unique counts
    profile_results["data_types"] = df.dtypes.astype(str).to_dict()
    profile_results["unique_counts"] = {col: int(df[col].nunique()) for col in df.columns}

    # Numeric summary
    profile_results["numeric_summary"] = df.describe(include=[np.number]).to_dict()

    # Column formatting issues
    col_formatting = {}
    for col in original_columns:
        
        # Only checking columns that are in the dataset after shielding/consideration
        if col in df.columns:
            issues = []
            
            if col.strip() != col:
                issues.append("has leading/trailing whitespace")
            
            if col != col.lower():
                issues.append("is not lower-case")
            
            col_formatting[col] = issues if issues else ["OK"]
        
        else:
            col_formatting[col] = ["Shielded or not considered"]
    
    # Check for incorrect values in 'state'-like columns
    state_flags = {}
    for col in df.columns:
        if "state" in col.lower():
            # Flag values that are not valid US states (assuming abbreviations)
            invalids = df[col].dropna().apply(lambda x: str(x).strip().upper() not in US_STATES).sum()
            state_flags[col] = int(invalids)
            
    profile_results["state_value_flags"] = state_flags

    # Record shielded columns (if any)
    profile_results["shielded_columns"] = shielded
    
    # User inputs
    profile_results["intended_use"] = intended_use
    profile_results["ai_system"] = ai_system
    profile_results["training_testing_ft"] = training_testing_ft
    
    print("=== Profile Results ===")
    pretty_print_json(profile_results)
    
    # Explanatory messages for each metric
    messages = {
        "shape": "Dataset dimensions (rows, columns).",
        "missing_values": "Missing values per column. Imputation or removal may be needed.",
        "duplicate_rows": "Duplicate rows count. Remove duplicates to avoid bias.",
        "data_types": "Data types per column. Consistency is key for training.",
        "unique_counts": "Unique entry counts per column.",
        "numeric_summary": "Statistics for numeric columns (mean, std, min, max, etc.).",
        "column_formatting": "Notes on column name formatting.",
        "state_value_flags": "Number of values in state columns not matching valid US state abbreviations.",
        "shielded_columns": "Columns shielded from profiling and assessment.",
        "intended_use": "The intended use of the dataset as provided by the user.",
        "ai_system": "The target AI system for which this dataset is intended.",
        "training_testing_ft": "The specific purpose this dataset is to be used for. Is it for training, testing, fine-tuning or neither?"
    }
    
    # Adding some more keys and values to the dictionary
    profile_results["column_formatting"] = col_formatting    
    profile_results["messages"] = messages
    return profile_results

def format_assessment_response(model_response):
    """
    Extracts and formats the assessment portion from the model's generated text.
    
    Parameters:
        model_response (list): A list of dictionaries from the model output.    
    Returns:
        str: The formatted assessment text.
    """
    
    if not model_response:
        return "No response received from model."
    
    raw_text = model_response[0].get("generated_text", "")
    keyword = "AI-readiness assessment:"
    
    if keyword in raw_text:
        
        # Extract everything after the keyword.
        assessment_text = raw_text.split(keyword, 1)[1].strip()
        return assessment_text
    
    else:
        # If the keyword is not found, return the full text.
        return raw_text

# Model API Call: Assessment and Fix Suggestions
# Using Hugging Face inference API
def get_model_response(model, prompt):
    print("Model: ", model)
    
    try:
        if model == "llama": # llama 3, 8B Parameters Model
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        
        elif model == "deepseek":
            API_URL = "https://api-inference.huggingface.co/models/unsloth/DeepSeek-R1-Distill-Llama-8B"
        
        else:
            raise ValueError(f"Model '{model}' is not supported. Please either choose llama or deepseek")
        
        headers = {"Authorization": f"Bearer "}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 2000}}

        response = requests.post(API_URL, headers=headers, json=payload)
        with open("llama_results", 'w') as file:
            json.dump(response.json(), file, indent=4)
            
        return format_assessment_response(response.json())
    
    except Exception as e:
        return f"Error contacting model API: {e}"    

def get_assessment_from_model(model, profile_results):
    """
    Curate a prompt from the profiling data and query the chosen model for an assessment.
    """
    prompt = (
        "Based on the following dataset profile (which is  JSON that includes information about the different columns of the dataset), assess its AI-readiness (while also taking into account its intended use). "
        "Highlight any issues and suggest actionable improvements to data quality,"
        "structure, and alignment with intended AI use cases. Be concise, clear, coherent. Make sure to start your response by saying: 'AI-readiness assessment:'\n\n"
        f"{json.dumps(profile_results, indent=2)}"
    )
    
    return get_model_response(model, prompt)

def get_fix_suggestion(model, old_profile, new_profile, old_assessment, new_assessment):
    """
    Given the profile before and after hard-coded fixes and their assessments, query the model
    for code suggestions and reasoning to further improve the dataset.
    """
    prompt = (
        "The dataset was profiled and assessed before and after applying basic fixes. "
        "Before fixes:\n"
        f"{json.dumps(old_profile, indent=2)}\n"
        f"Assessment: {old_assessment}\n\n"
        "After basic fixes:\n"
        f"{json.dumps(new_profile, indent=2)}\n"
        f"Assessment: {new_assessment}\n\n"
        "Generate a concise Python code snippet that would further fix and optimize the dataset, "
        "and explain why this fix is needed."
    )
    
    return get_model_response(model, prompt)

# Data Assessment (AI-Readiness)
def assess_data(file_path, consider_cols=None, shield_cols=None, model="llama"):
    """
    Loads cached profiling results (or profiles the dataset) and then gets an assessment
    from the selected model.
    """
    profile_results = profile_data(file_path, consider_cols, shield_cols)

    # Local assessments based on the profile
    suggestions = []
    shape = profile_results.get("shape", (0, 0))
    
    if shape[0] == 0:
        suggestions.append("Dataset is empty. Ensure data availability.")
    else:
        suggestions.append(f"Dataset has {shape[0]} rows and {shape[1]} columns.")

    # Missing values
    total_missing = sum(profile_results.get("missing_values", {}).values())
    suggestions.append("Missing values found." if total_missing > 0 else "No missing values detected.")

    for col, issues in profile_results.get("column_formatting", {}).items():
        if issues != ["OK"] and "Shielded" not in issues[0]:
            suggestions.append(f"Column '{col}' issues: {', '.join(issues)}")

    # Duplicates
    duplicates = profile_results.get("duplicate_rows", 0)
    suggestions.append(f"{duplicates} duplicate rows found." if duplicates > 0 else "No duplicate rows detected.")

    numeric_features = [col for col, dtype in profile_results.get("data_types", {}).items() if "int" in dtype or "float" in dtype]
    suggestions.append(f"Numeric features: {numeric_features}" if numeric_features else "No numeric features detected.")

    state_flags = profile_results.get("state_value_flags", {})
    for col, count in state_flags.items():
        if count > 0:
            suggestions.append(f"Column '{col}' has {count} values not matching valid US state abbreviations.")

    print("=== Local Assessment Suggestions ===")
    for s in suggestions:
        print("- " + s)

    # Query the model for an advanced assessment (minimized call)
    print("\nRequesting advanced assessment from model...")
    model_assessment = get_assessment_from_model(model, profile_results)
    print("\n=== Model Assessment ===")
    print(model_assessment)
    return model_assessment

# Sensitive Data Masking
def mask_data(file_path, delay_minutes, output_file, mask_cols=None):
    """
    Mask sensitive data.
    - If mask_cols is specified, only those columns are masked.
    - Otherwise, columns with sensitive keywords are auto-detected.
    """
    sensitive_keywords = ['email', 'ssn', 'password', 'credit', 'card']
    
    try:
        df = pd.read_csv(file_path)
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # If user specified mask_cols, verify they exist.
    user_mask_cols = []
    if mask_cols:
        user_mask_cols = [col.strip() for col in mask_cols.split(",")]
        
        for col in user_mask_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' specified for masking does not exist.")
    
    masked_columns = []
    backup_data = {}

    # Decide which columns to mask
    for col in df.columns:
        
        # Use user specification if provided; else, auto-detect sensitive columns.
        if user_mask_cols and col in user_mask_cols:
            to_mask = True
        elif not user_mask_cols and any(keyword in col.lower() for keyword in sensitive_keywords):
            to_mask = True
        else:
            to_mask = False

        if to_mask:
            masked_columns.append(col)
            backup_data[col] = df[col].to_dict()
            df[col] = "****"

    df.to_csv(output_file, index=False)
    print(f"Masked CSV saved to {output_file}")

    metadata = {
        "mask_time": datetime.now().isoformat(),
        "delay_minutes": delay_minutes,
        "masked_columns": masked_columns
    }
    
    with open("mask_metadata.json", "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)
    
    with open("mask_backup.json", "w") as backup_file:
        json.dump(backup_data, backup_file, indent=4)
    
    print("Metadata and backup of sensitive data saved.")

    # Re-profile to update cached profile
    profile_data(file_path)

# Auto-Fix Workflow
def fix_data(file_path, output_file, consider_cols=None, shield_cols=None, model="llama"):
    """
    Applies hard-coded fixes, then re-profiles and re-assesses the dataset.
    Finally, queries the model to generate additional code suggestions to further fix the dataset.
    The model's code suggestion (with reasoning) is printed for user review.
    """
    # Profile before fixes
    old_profile = profile_data(file_path, consider_cols, shield_cols)
    old_assessment = assess_data(file_path, consider_cols, shield_cols, model)

    # Basic hard-coded fixes: clean column names and impute missing values.
    try:
        df = pd.read_csv(file_path)
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Impute missing values: numeric with median; others with mode.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        
        else:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna("", inplace=True)
    
    # Save basic fixed data
    fixed_basic_file = "data_basic_fixed.csv"
    df.to_csv(fixed_basic_file, index=False)
    print(f"Basic fixed CSV saved to {fixed_basic_file}")

    # Re-profile the fixed data
    new_profile = profile_data(fixed_basic_file, consider_cols, shield_cols)
    new_assessment = assess_data(fixed_basic_file, consider_cols, shield_cols, model)

    # Get model suggestions for further improvements
    print("\nRequesting further fix suggestions from model...")
    fix_suggestion = get_fix_suggestion(model, old_profile, new_profile, old_assessment, new_assessment)
    print("\n=== Model Fix Suggestion ===")
    print(fix_suggestion)

    print("\nFix workflow complete. Review the above code suggestion to apply additional improvements.")

# Main CLI Entry Point
def main():
    parser = argparse.ArgumentParser(
        description="Ready4AI: Check your dataset's AI-readiness by profiling, assessing, masking, fixing, and handling sensitive data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands: profile, assess, mask, release, fix")

    # Global optional parameters for profiling/assessment (columns)
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--consider-cols", help="Comma-separated list of columns to consider for profiling/assessment")
    common_parser.add_argument("--shield-cols", help="Comma-separated list of columns to shield (exclude from profiling)")

    # Profile command
    profile_parser = subparsers.add_parser("profile", parents=[common_parser], help="Profile CSV data structure and quality")
    profile_parser.add_argument("--input", required=True, help="Input CSV file")

    # Assess command
    assess_parser = subparsers.add_parser("assess", parents=[common_parser], help="Assess dataset AI-readiness")
    assess_parser.add_argument("--input", required=True, help="Input CSV file")
    assess_parser.add_argument("--model", default="llama", help="Model to use for assessment (llama or deepseek)")

    # Mask command
    mask_parser = subparsers.add_parser("mask", help="Mask sensitive data in the CSV")
    mask_parser.add_argument("--input", required=True, help="Input CSV file")
    mask_parser.add_argument("--output", default="data_masked.csv", help="Output CSV file for masked data")
    mask_parser.add_argument("--mask-cols", help="Comma-separated list of columns to mask (overrides auto-detection)")

    # Fix command
    fix_parser = subparsers.add_parser("fix", parents=[common_parser], help="Fix dataset formatting issues and get model fix suggestions")
    fix_parser.add_argument("--input", required=True, help="Input CSV file")
    fix_parser.add_argument("--output", default="data_fixed.csv", help="Output CSV file for fixed data")
    fix_parser.add_argument("--model", default="llama", help="Model to use for fix suggestions (llama or deepseek)")

    args = parser.parse_args()

    if args.command == "profile":
        profile_data(args.input, args.consider_cols, args.shield_cols)
    
    elif args.command == "assess":
        assess_data(args.input, args.consider_cols, args.shield_cols, args.model)
    
    elif args.command == "mask":
        mask_data(args.input, args.output, args.mask_cols)
    
    elif args.command == "fix":
        fix_data(args.input, args.output, args.consider_cols, args.shield_cols, args.model)

if __name__ == "__main__":
    main()