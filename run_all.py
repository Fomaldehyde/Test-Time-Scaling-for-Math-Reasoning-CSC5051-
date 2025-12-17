import os
import json
from config import (
    DATASET_SPLIT, SAMPLE_NUM, MAX_NEW_TOKENS_BASE, MAX_NEW_TOKENS_LONG,
    TEMPERATURE_BASE, NUM_SAMPLES
)
from data.data_loader import load_math_dataset, sample_questions
from src.prompts import get_prompt
from src.generate import generate_and_save_answers
from src.evaluate import evaluate_passk, self_consistency_passk

# Allow overriding sample count from environment; falls back to config.NUM_SAMPLES
RUN_NUM_SAMPLES = int(os.getenv("RUN_NUM_SAMPLES", NUM_SAMPLES))

def main():
    # Step 1: 加载测试数据集
    print("="*50 + " Load Dataset " + "="*50)
    questions = load_math_dataset(split=DATASET_SPLIT, n=None)
    questions_to_test = sample_questions(questions, num_samples=SAMPLE_NUM)
    
    # Step 2: 定义实验配置
    experiments = [
        {
            "method_name": "base",
            "prompt_type": "base_only_answer",
            "num_samples": RUN_NUM_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS_BASE
        },
        {
            "method_name": "base",
            "prompt_type": "base_only_answer",
            "num_samples": 3,
            "max_new_tokens": MAX_NEW_TOKENS_BASE
        },
        # {
        #     "method_name": "0_shot",
        #     "prompt_type": "0_shot",
        #     "num_samples": RUN_NUM_SAMPLES,
        #     "max_new_tokens": MAX_NEW_TOKENS_BASE
        # },
        # {
        #     "method_name": "cot_detailed",
        #     "prompt_type": "cot_detailed",
        #     "num_samples": RUN_NUM_SAMPLES,
        #     "max_new_tokens": MAX_NEW_TOKENS_LONG
        # },
        # {
        #     "method_name": "few_shot",
        #     "prompt_type": "few_shot",
        #     "do_sample": False,
        #     "max_new_tokens": MAX_NEW_TOKENS_LONG
        # },
        # {
        #     "method_name": "cot_check",
        #     "prompt_type": "cot_check",
        #     "temperature": TEMPERATURE_BASE,
        #     "max_new_tokens": MAX_NEW_TOKENS_COT
        # },
        # {
        #     "method_name": "mini_cot",
        #     "prompt_type": "mini_cot",
        #     "do_sample": False,
        #     "max_new_tokens": MAX_NEW_TOKENS_LONG
        # },
        # {
        # "method_name": "base_with_reflection",
        # "prompt_type": "base_only_answer",
        # "temperature": TEMPERATURE_BASE,
        # "max_new_tokens": 4096,
        # "use_reflection": True  
        # }
    ]
    
    # Step 3: 运行所有实验（生成答案）
    print("\n" + "="*50 + " Run Experiments " + "="*50)
    result_files = []
    for exp in experiments:
        prompt = get_prompt(exp["prompt_type"])
        # Per-experiment samples if provided; otherwise fall back to RUN_NUM_SAMPLES
        exp_num_samples = int(exp.get("num_samples", RUN_NUM_SAMPLES))
        out_file = generate_and_save_answers(
            questions_to_test=questions_to_test,
            method_name=exp["method_name"],
            prompt=prompt,
            num_samples=exp_num_samples,
            # If only 1 sample, run deterministically; else enable sampling
            do_sample=(exp_num_samples > 1),
            # Use configured temperature when sampling; ignored if do_sample=False
            temperature=exp.get("temperature", TEMPERATURE_BASE),
            max_new_tokens=exp["max_new_tokens"]
        )
        result_files.append((exp["method_name"], out_file, exp_num_samples))
    
    # Step 4: 评估所有实验结果（Pass@k）
    # Step 4: Evaluate all experimental results (Pass@k + Latency + Token)
    print("\n" + "="*50 + " Evaluate Results " + "="*50)
    all_reports = {}
    for method_name, jsonl_path, exp_num_samples in result_files:
        report = evaluate_passk(jsonl_path, pass_k=exp_num_samples)
        all_reports[method_name] = report
        
        # Print core metrics (English only, no emoji)
        print(f"\n=== {method_name} ===")
        print(f"Core Pass Rate:")
        print(f"   - Total questions: {report['total']}")
        print(f"   - Pass@{exp_num_samples}: {report['pass@k']:.4f} ({report['pass@k']*100:.2f}%)")
        
        print(f"\nLatency Statistics (Unit: seconds):")
        print(f"   - Average latency per question: {report['latency_avg']:.2f}")
        print(f"   - Total generation latency: {report['latency_total']:.2f}")
        
        print(f"\nToken Statistics:")
        print(f"   - Total prompt tokens: {report['prompt_tokens_total']}")
        print(f"   - Total completion tokens: {report['completion_tokens_total']}")
        # Optional: Total tokens & average tokens per question
        total_tokens = report['prompt_tokens_total'] + report['completion_tokens_total']
        print(f"   - Total token consumption: {total_tokens}")
        print(f"   - Average tokens per question: {total_tokens / report['total']:.0f}")
        
    # Step 5: 保存最终评估报告
    with open(f"{os.getenv('SAVE_DIR', './experiment_results')}/final_report.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print("\nFinal report saved to → experiment_results/final_report.json")

if __name__ == "__main__":
    # 检查API密钥是否设置
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set!")
        print("Please set it first (e.g., export OPENAI_API_KEY='sk-xxx' on Linux/Mac, set OPENAI_API_KEY=sk-xxx on Windows)")
        exit(1)
    main()