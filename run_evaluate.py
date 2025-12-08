import os
import sys
import json
import glob
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# ========== 正确导入 ==========
from src.evaluate import evaluate_passk, self_consistency_passk
from config import NUM_SAMPLES

# ========== 配置项 ==========
TARGET_DIR = "./experiment_results"
SAVE_SUMMARY = True
SUMMARY_SAVE_PATH = os.path.join(TARGET_DIR, "batch_evaluation_summary.json")

def batch_evaluate_jsonl():
    # 1. Check target directory
    if not os.path.exists(TARGET_DIR):
        print(f"[ERROR] Target directory {TARGET_DIR} does not exist!")
        return
    
    # 2. Scan all raw jsonl files
    jsonl_files = glob.glob(os.path.join(TARGET_DIR, "**/raw*.jsonl"), recursive=True)
    if not jsonl_files:
        print(f"[WARNING] No .jsonl files found in {TARGET_DIR}!")
        return
    
    # 3. Initialize summary variables
    summary_results = {}
    total_passk = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_total = 0.0
    total_latency_avg = 0.0
    success_count = 0
    fail_count = 0

    # 4. Batch evaluation
    print("="*80)
    print(f"[EVALUATION] Start batch evaluation (found {len(jsonl_files)} jsonl files)")
    print("="*80)
    
    for idx, file_path in enumerate(jsonl_files, 1):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        print(f"\n[{idx}/{len(jsonl_files)}] Evaluating file: {file_name} (size: {file_size:.2f} KB)")
        print("-" * 60)
        
        try:
            # Extract pass@k from filename
            k_match = re.search(r"pass@(\d+)", file_name)
            pass_k = int(k_match.group(1)) if k_match else NUM_SAMPLES
            
            eval_result = evaluate_passk(
                jsonl_path=file_path,
                pass_k=pass_k
            )
            
            # Print single file metrics
            print(f"[METRICS] Single file metrics:")
            print(f"   - Total questions: {eval_result['total']}")
            print(f"   - Pass@{pass_k}: {eval_result['pass@k']:.4f} ({eval_result['pass@k']*100:.2f}%)")
            print(f"   - Avg latency per question: {eval_result['latency_avg']:.2f} sec")
            print(f"   - Total latency: {eval_result['latency_total']:.2f} sec")
            print(f"   - Total prompt tokens: {eval_result['prompt_tokens_total']}")
            print(f"   - Total completion tokens: {eval_result['completion_tokens_total']}")
            
            # Accumulate statistics
            total_passk += eval_result['pass@k']
            total_prompt_tokens += eval_result['prompt_tokens_total']
            total_completion_tokens += eval_result['completion_tokens_total']
            total_latency_total += eval_result['latency_total']
            total_latency_avg += eval_result['latency_avg']
            success_count += 1
            
            # Save to summary
            summary_results[file_name] = eval_result
            
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {str(e)}")
            summary_results[file_name] = {"error": str(e)}
            fail_count += 1

    # 5. Save summary results
    if SAVE_SUMMARY and summary_results:
        with open(SUMMARY_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        print(f"\n[SUCCESS] Summary saved to: {SUMMARY_SAVE_PATH}")

    # 6. Print final summary
    print("\n" + "="*80)
    print("[SUMMARY] Batch evaluation final summary")
    print("="*80)
    print(f"[SUMMARY] Successfully evaluated files: {success_count}")
    print(f"[SUMMARY] Failed evaluated files: {fail_count}")
    print(f"[SUMMARY] Total scanned files: {len(jsonl_files)}")
    

if __name__ == "__main__":
    # Optional: Set console encoding to UTF-8 (Windows)
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except:
            pass
    batch_evaluate_jsonl()