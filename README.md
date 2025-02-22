# Ready4AI

## Overview
Ready4AI is a command-line tool designed to assess and improve the AI-readiness of datasets, specifically CSV files. It performs comprehensive data profiling, identifies quality issues, masks sensitive information, and offers actionable suggestions to align datasets with AI use cases, such as model training, testing, and fine-tuning.

### Key AI-Readiness Factors:
- **Data Availability:** Ensures datasets have sufficient data for AI applications.
- **Data Quality:** Identifies missing values, formatting issues, duplicates, and incorrect data.
- **Proper Structure:** Validates column names, data types, and granularity.
- **Alignment with AI Use Cases:** Aligns datasets with intended AI tasks, such as computer vision, NLP, or predictive modeling.

## Features

### 1. Command-Line Interface (CLI)
- Simple and flexible CLI for dataset profiling, assessment, masking, and fixing.
- Supports optional parameters to customize profiling, such as column selection and shielding.
- Outputs results to the terminal and saves key information to files.

### 2. Data Profiling
- Provides dataset dimensions, missing values, duplicate counts, data types, and unique counts.
- Generates numeric summaries (mean, median, standard deviation, min, max) for numeric columns.
- Identifies formatting issues (whitespace, lowercase inconsistencies).
- Flags invalid US state abbreviations.

### 3. AI-Readiness Assessment
- Uses open-source AI models (Llama 3 8B and DeepSeek-R1) to assess dataset quality.
- Offers concise, coherent assessments highlighting data alignment with AI use cases.
- Suggests actionable improvements to data structure and quality.

### 4. Sensitive Data Detection & Masking
- Auto-detects sensitive columns based on keywords like "email," "password," and "credit."  
- Masks sensitive values with "****" in the output file.
- Allows manual specification of columns to mask.

### 5. Automated Data Fixing
- Cleans column names (trims whitespace, converts to lowercase).
- Imputes missing numeric values with medians and categorical values with modes.
- Saves fixed datasets and re-assesses their AI-readiness.
- Queries AI models for additional code suggestions to further improve data quality.

---

## Installation

### Requirements
- Python 3.x
- Pandas
- NumPy
- Requests

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Ready4AI.git
cd Ready4AI

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### CLI Commands
```bash
# Profile a dataset
python ready4ai.py profile --input combined_monthly_dataset.csv

# Assess AI-readiness
python ready4ai.py assess --input combined_monthly_dataset.csv --model llama

# Mask sensitive columns
python ready4ai.py mask --input TestReady.csv --output data_masked.csv --mask-cols email,ssn

# Fix dataset formatting and get AI model suggestions
python ready4ai.py fix --input combined_monthly_dataset.csv --output data_fixed.csv --model deepseek
```

### Optional Parameters
- `--consider-cols`: Comma-separated list of columns to include in profiling.
- `--shield-cols`: Comma-separated list of columns to exclude from profiling.
- `--mask-cols`: Comma-separated list of columns to mask.
- `--model`: AI model to use (choose "llama" or "deepseek").

---

## File Outputs
- **Profile Results:** Saved as `profile_results.json`
- **Masked Data:** Saved as `data_masked.csv` with original values backed up in `mask_backup.json`
- **Fixed Data:** Saved as `data_fixed.csv` and `data_basic_fixed.csv`
- **AI Assessments:** Printed in the terminal and saved in `llama_results`

---

## Example Workflow
1. **Profile the dataset:** Check dataset dimensions, missing values, and formatting issues.
2. **Assess AI-readiness:** Use AI models to evaluate dataset alignment with intended use cases.
3. **Mask sensitive data:** Protect sensitive columns before sharing the dataset.
4. **Fix common issues:** Automatically apply basic fixes and receive additional improvement suggestions.

---

## Contributing
Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
- Uses [Hugging Face Inference API](https://huggingface.co/) for AI assessments.
- Built with Pandas and NumPy for efficient data manipulation.

