# NLP Final Project: Math Problem Solving with Test-Time Scaling
This project evaluates different prompt strategies for math problem solving (GSM8K/AIME) using LLMs via OpenAI-compatible API.

## Environment Setup
### 1. Clone the repo
```bash
git clone <your-github-repo-url>
cd <repo-name>
```
### 2. Create virtual environment (optional but recommended)
```
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Set API key (critical!)
```
# Linux/Mac
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"  # optional, default is SiliconFlow

# Windows (Command Prompt)
set OPENAI_API_KEY="your-api-key"
set OPENAI_API_BASE="https://api.siliconflow.cn/v1"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_API_BASE="https://api.siliconflow.cn/v1"
```
### Run Experiments

```
python run_all.py
```

### Directory Structure
```
├── README.md               # This file
├── requirements.txt        # Dependencies
├── config.py               # Global config (API/model/path)
├── run_all.py              # Main script (one-click run)
├── run_evaluate.py         # evaluate based files in /experiment_results.
├── data/
│   └── data_loader.py      # Dataset loading (GSM8K/MATH-500)
└── src/
    ├── model_api.py        # LLM API call
    ├── evaluate.py         # Answer normalization & evaluation
    ├── experiment.py       # Experiment pipeline (generate/evaluate)
    └── prompts.py          # Prompt templates (CoT/few-shot/etc.)
```