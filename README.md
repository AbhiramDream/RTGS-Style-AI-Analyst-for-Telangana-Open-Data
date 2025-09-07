#  RTGS-Style AI Analyst for Telangana Open Data

An AI-powered pipeline to analyze **Telangana Open Data** in real-time.  
This tool enables users to load CSV datasets, clean and impute missing values, run analysis, and query them in natural language using AI models.  
It is designed for researchers, policymakers, and data enthusiasts to quickly explore and interpret government data.

---
#Demo video
https://drive.google.com/file/d/1aX3px4-1JMY9nj488ggZ39mXlTn3zaxZ/view?usp=sharing
##  Features
- Load and preprocess Telangana Open Data
- Clean and impute missing values
- Analyze datasets with AI-assisted insights
- Query datasets using natural language
- Interactive CLI for conversational analysis
- Multiple AI backends (Ollama)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AbhiramDream/RTGS-Style-AI-Analyst-for-Telangana-Open-Data.git
cd RTGS-Style-AI-Analyst-for-Telangana-Open-Data/pipeline

# Create and activate virtual environment
python -m venv TBAI
TBAI\Scripts\activate   # On Windows
source TBAI/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Load a dataset
python -m pipeline.cli_interface load "path/to/data.csv"

# 2. Clean dataset
python -m pipeline.cli_interface clean

# 3. Impute missing values
python -m pipeline.cli_interface impute

# 4. Run analysis
python -m pipeline.cli_interface analyze

# 5. Query with natural language
python -m pipeline.cli_interface --query "Show top 5 districts by diagnostics count"

# 6. Run in interactive mode
python -m pipeline.cli_interface --interactive

# 7. Show help
python -m pipeline.cli_interface -h
```

---

## 🔄 Example Workflow

```bash
# Step 1: Load dataset
python -m pipeline.cli_interface load Telangana_Diagnostics_Data_2024_08.csv

# Step 2: Clean data
python -m pipeline.cli_interface clean

# Step 3: Impute missing values
python -m pipeline.cli_interface impute

# Step 4: Analyze dataset
python -m pipeline.cli_interface analyze

# Step 5: Query insights
python -m pipeline.cli_interface --query "Top 10 hospitals by patient count"
```

---

##  Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

---

