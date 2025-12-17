# Test-Time Scaling for Math Reasoning
Solving [MATH problems](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) using LLMs(qwen8b).

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

## Usage

### Run All Experiments
Run the complete experiment pipeline (generation + evaluation):
```bash
python run_all.py
```

### Apply Reflection to Existing Results
Post-process previously generated results with self-correction:
```bash
python reflection.py
```
Edit the `__main__` section in `reflection.py` to specify input/output files.

### Evaluate Only
Evaluate existing result files in `/experiment_results`:
```bash
python run_evaluate.py
```
By default, the project uses `evaluate_passk` (Pass@k metric) to evaluate results — it checks if any of the k reasoning paths is correct. To get self-consistency results (majority voting on k paths), replace `evaluate_passk` with `self_consistency_passk` 

### Directory Structure
```
├── README.md               # This file
├── requirements.txt        # Dependencies
├── config.py               # Global config (API/model/path)
├── run_all.py              # Main script (one-click run)
├── run_evaluate.py         # evaluate based files in /experiment_results.
├── reflection.py           # Post-process existing results with reflection/self-correction
├── data/
│   └── data_loader.py      # Dataset loading (MATH-500)
└── src/
    ├── model_api.py        # LLM API call
    ├── evaluate.py         # Answer normalization & evaluation
    ├── generate.py         # Answer generation with optional reflection
    ├── grade.py            # Grading logic
    └── prompts.py          # Prompt templates (CoT/few-shot/etc.)
```