import os
import json
from config import (
    DATASET_SPLIT, SAMPLE_NUM, MAX_NEW_TOKENS_BASE, MAX_NEW_TOKENS_COT,
    TEMPERATURE_BASE, TEMPERATURE_COT, NUM_SAMPLES
)
from data.data_loader import load_math_dataset, sample_questions
from src.prompts import get_prompt
from src.experiment import generate_and_save_answers, evaluate_passk

def main():
    # Step 1: 加载测试数据集
    print("="*50 + " Load Dataset " + "="*50)
    questions = load_math_dataset(split=DATASET_SPLIT, n=None)
    questions_to_test = sample_questions(questions, num_samples=SAMPLE_NUM)
    
    # Step 2: 定义所有实验配置
    experiments = [
        # {
        #     "method_name": "base_empty_prompt",
        #     "prompt_type": "base_empty",
        #     "temperature": 0.3,
        #     "max_new_tokens": MAX_NEW_TOKENS_BASE
        # },
        # {
        #     "method_name": "cot_detailed",
        #     "prompt_type": "cot_detailed",
        #     "temperature": TEMPERATURE_COT,
        #     "max_new_tokens": MAX_NEW_TOKENS_COT
        # },
        # {
        #     "method_name": "base",
        #     "prompt_type": "base_only_answer",
        #     "temperature": TEMPERATURE_BASE,
        #     "max_new_tokens": MAX_NEW_TOKENS_BASE
        # },
        # {
        #     "method_name": "cot_check",
        #     "prompt_type": "cot_check",
        #     "temperature": TEMPERATURE_COT,
        #     "max_new_tokens": MAX_NEW_TOKENS_COT
        # },
        # {
        #     "method_name": "few_shot",
        #     "prompt_type": "few_shot",
        #     "temperature": TEMPERATURE_COT,
        #     "max_new_tokens": MAX_NEW_TOKENS_COT
        # },
        # {
        #     "method_name": "mini_cot",
        #     "prompt_type": "mini_cot",
        #     "temperature": TEMPERATURE_COT,
        #     "max_new_tokens": MAX_NEW_TOKENS_COT
        # },
        {
        "method_name": "cot_detailed_with_reflection",
        "prompt_type": "cot_detailed",
        "temperature": TEMPERATURE_COT,
        "max_new_tokens": 4096,
        "use_reflection": True  # 启用反思
        }
    ]
    
    # Step 3: 运行所有实验（生成答案）
    print("\n" + "="*50 + " Run Experiments " + "="*50)
    result_files = []
    for exp in experiments:
        prompt = get_prompt(exp["prompt_type"])
        out_file = generate_and_save_answers(
            questions_to_test=questions_to_test,
            method_name=exp["method_name"],
            prompt=prompt,
            num_samples=NUM_SAMPLES,
            temperature=exp["temperature"],
            max_new_tokens=exp["max_new_tokens"]
        )
        result_files.append((exp["method_name"], out_file))
    
    # Step 4: 评估所有实验结果（Pass@k）
    # Step 4: Evaluate all experimental results (Pass@k + Latency + Token)
    print("\n" + "="*50 + " Evaluate Results " + "="*50)
    all_reports = {}
    for method_name, jsonl_path in result_files:
        report = evaluate_passk(jsonl_path, pass_k=NUM_SAMPLES)
        all_reports[method_name] = report
        
        # Print core metrics (English only, no emoji)
        print(f"\n=== {method_name} ===")
        print(f"Core Pass Rate:")
        print(f"   - Total questions: {report['total']}")
        print(f"   - Pass@{NUM_SAMPLES}: {report['pass@k']:.4f} ({report['pass@k']*100:.2f}%)")
        
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