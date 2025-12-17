# Test-Time Scaling for Math Reasoning

Exploring test-time scaling strategies for mathematical reasoning using Large Language Models. This project evaluates multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought) on the [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) using Qwen-8B model.

## Features

- Multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought)
- Self-consistency via majority voting across k samples
- Optional self-reflection for answer refinement
- Configurable random seed for reproducibility
- Per-question detailed analysis export
- Automatic checkpoint and resume support

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

## Configuration

Edit `config.py` to customize:
- `MODEL_NAME`: LLM model (default: "Qwen/Qwen3-8B")
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `NUM_SAMPLES`: Number of samples to generate per question (default: 1)
- `SAVE_DIR`: Output directory for results (default: "./experiment_results")

**Environment Variables** (override config.py):
```bash
# Override number of samples at runtime
RUN_NUM_SAMPLES=3 python run_all.py

# Enable detailed output for specific file
DETAIL_JSONL=raw_base_pass@1_seed42.jsonl python run_detail.py

# Export all detailed results
DETAIL_ALL=1 python run_detail.py
```

## Usage

### 1. Run All Experiments
Run the complete experiment pipeline (generation + evaluation):
```bash
python run_all.py
```

This will:
- Generate k samples per question using configured prompting strategies
- Evaluate results using self-consistency (majority voting)
- Save raw outputs to `experiment_results/raw_*.jsonl`
- Generate summary report in `experiment_results/batch_evaluation_summary.json`

**Output filename format**: `raw_{method}_pass@{k}_seed{seed}.jsonl` (k = number of samples)

### 2. Export Detailed Per-Question Results
Export detailed correctness analysis for each question:
```bash
# Export details for all raw*.jsonl files
python run_detail.py

# Export for specific file
DETAIL_JSONL=raw_base_pass@1_seed42.jsonl python run_detail.py
```

Output: `experiment_results/detailed/{method}_pass@{k}_seed{seed}_detailed.jsonl`

Each line contains:
- `question_id`: Question identifier
- `reference_answer`: Ground truth answer
- `sample_answers`: All k generated answers
- `model_answer`: Final answer after majority voting (when k>1) or single answer (when k=1)
- `is_correct`: Boolean correctness

### 3. Apply Reflection to Existing Results
Post-process previously generated results with self-correction:
```bash
python reflection.py
```
Edit the `__main__` section in `reflection.py` to specify input/output files.

### 4. Evaluate Only
Evaluate existing result files in `experiment_results/`:
```bash
python run_evaluate.py
```

**Evaluation Modes**:
- `self_consistency_passk`: **Primary method** - Majority voting across k samples for robust answer selection
- `evaluate_passk`: Alternative - Pass@k metric (checks if any of k samples is correct)

## Evaluation Strategy

This project primarily uses **self-consistency** (majority voting) for test-time scaling:

**When k=1** (single sample):
- Generate one answer per question
- Direct evaluation against ground truth

**When k>1** (multiple samples):
- Generate k different reasoning paths using `temperature=0.7`
- Apply majority voting: the most frequent answer among k samples becomes the final answer
- This leverages diversity in reasoning to improve robustness

The same generated samples support both evaluation modes:
- **Self-Consistency** (default): Vote on k samples to get the most reliable answer
- **Pass@k** (optional): Check if any of the k samples is correct (more lenient metric)

## Experiment Configuration

Experiments are defined in `run_all.py`. Each experiment has:
- `method_name`: Identifier for the prompting strategy
- `use_reflection`: Enable self-correction (True/False)
- `num_samples`: Override default NUM_SAMPLES (optional)

Example experiment configuration:
```python
experiments = [
    {"method_name": "0_shot", "use_reflection": False},
    {"method_name": "cot_detailed", "use_reflection": False, "num_samples": 3},
    {"method_name": "few_shot", "use_reflection": True, "num_samples": 5},
]
```

**Automatic behavior**:
- When `num_samples=1`: Uses `do_sample=False` (deterministic)
- When `num_samples>1`: Uses `do_sample=True` with `temperature=0.7`

## Project Structure
```
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── config.py               # Global configuration
├── run_all.py              # Main experiment pipeline
├── run_evaluate.py         # Standalone evaluation script
├── run_detail.py           # Export per-question detailed results
├── reflection.py           # Post-process with self-reflection
├── data/
│   └── data_loader.py      # MATH-500 dataset loader
└── src/
    ├── model_api.py        # LLM API interface (SiliconFlow)
    ├── evaluate.py         # Evaluation metrics (self-consistency, Pass@k)
    ├── generate.py         # Answer generation with resume support
    ├── grade.py            # Answer extraction and grading
    └── prompts.py          # Prompt templates library
```

## Output Files

### Raw Results
`experiment_results/raw_{method}_pass@{k}_seed{seed}.jsonl`
- Each line: one question with k model outputs
- Fields: `question_id`, `question`, `reference_answer`, `model_outputs`, `total_prompt_tokens`, `total_completion_tokens`, `total_latency`, `config`

### Detailed Analysis
`experiment_results/detailed/{method}_pass@{k}_seed{seed}_detailed.jsonl`
- Each line: per-question correctness details
- Fields: `question_id`, `reference_answer`, `sample_answers`, `model_answer`, `is_correct`
- `model_answer` is determined by majority voting when k>1

### Evaluation Summary
`experiment_results/batch_evaluation_summary*.json`
- Aggregated metrics across all experiments
- Fields: `method_name`, `accuracy`, `correct`, `total`, `pass_k` (or `k` for self-consistency)

## Prompting Strategies

Available in `src/prompts.py`:
- **base**: Direct question answering
- **0_shot**: Zero-shot prompting
- **few_shot**: Few-shot COT with examples
- **cot_detailed**: Detailed Chain-of-Thought
- **cot_check**: CoT with self-verification
- **mini_cot**: Concise CoT

## Answer Extraction

The system uses robust LaTeX answer extraction (`src/grade.py`):
- Extracts content from `\boxed{...}` with arbitrary nesting levels (brace-counting algorithm)
- Handles incomplete LaTeX expressions
- Removes unit markers (e.g., `\text{th}`, `\text{cents}`)
- Normalizes mathematical expressions using SymPy

## Requirements

- Python 3.8+
- openai >= 1.0.0
- sympy
- tqdm
- pylatexenc

See `requirements.txt` for complete list.

## License

This project is for academic purposes (CSC5051 NLP Final Project).

## Acknowledgments

- Dataset: [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- Model: Qwen-8B via [SiliconFlow API](https://siliconflow.cn/)