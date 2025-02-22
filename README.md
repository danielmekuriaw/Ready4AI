# Ready4AI
Checking AI-readiness: A command-line tool that ingests a CSV file, assesses data quality, flags/masks sensitive columns and checks the viability of a dataset for model training/testing/fine-tuning.

### What matters for AI-readiness (Link)[https://www2.deloitte.com/us/en/pages/advisory/articles/data-preparation-for-ai.html]
- Data availability
- Data quality
- Properly structured data
- Data aligned with its AI use cases

### Features

1. **Command-Line Interface:**
   - Accept an input CSV file and optional parameters.
   - Provide commands like `profile`, `mask`, and so on for different types of tasks.
   - Output results directly to the terminal or write to a new CSV/.txt file.

2. **Data Quality Assessment:**
   - **Basic Profiling:**  

     Compute simple statistics (e.g., missing values, data types, distributions, number of unique entries and so on) for each column (attributes that are useful to determine how useful a dataset could be to improve).
   - **Formatting Suggestions:**  

     By feeding the initial data quality assessment results to an open-source model, identify columns with obvious issues (like mixed data types), highlight them and suggest common fixes.
   - **Assessing Data Alignment with Intended Use Case:**
     
     Using the profiling results and what is returned from the open-source model to analyze to what extent it is aligned.

3. **Sensitive Data Detection & Masking:**
   - **Keyword-Based Flagging:**  

     Identify columns whose names contain common sensitive keywords (for instance, "email", "ssn", "password", "credit").
   - **Masking:**  

     Replace sensitive values with a mask (like "****") in the output.
   - **Shielding:**

     What is communicated to the models are just some useful/critical assessments of the dataset but not any specific data points. The data remains locally and shielded from the rest of the application.

4. **Complete Some of the Suggestions on its own:**
   - **Fixing**

     If the user chooses to, can specify the command so that the application addresses some of the issues that it can.
