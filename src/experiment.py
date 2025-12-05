import os
import json
import time
from tqdm import tqdm
from src.model_api import generate_answer_api, reflect_and_correct
from src.evaluate import grade_answer, extract_answer
from config import SAVE_DIR, NUM_SAMPLES


def generate_and_save_answers(
    questions_to_test,
    method_name="default",
    prompt_type="base_empty",
    prompt="",
    num_samples=NUM_SAMPLES,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024,
    use_reflection=False  # 新增：是否启用反思
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_file = f"{SAVE_DIR}/raw_{method_name}_pass@{num_samples}.jsonl"
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx, item in tqdm(enumerate(questions_to_test, 1),
                    total=len(questions_to_test),
                    desc=f"Generating {method_name}"):
            question = item["question"]
            ref_ans = item["answer"]
            samples = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_latency = 0.0
            
            for s in range(num_samples):
                t0 = time.time()
                # 1. 生成初始答案
                gen_text, p_tokens, c_tokens, _ = generate_answer_api(
                    question=question,
                    prompt=prompt,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens
                )
                
                # 2. 若启用反思，修正答案
                if use_reflection:
                    corrected_text = reflect_and_correct(
                        question=question,
                        original_output=gen_text
                    )
                    # 合并初始+修正输出（便于后续分析）
                    gen_text = f"Original: {gen_text}\nCorrected: {corrected_text}"
                
                latency = time.time() - t0
                # 累计统计
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                total_latency += latency
                # 保存单次样本
                samples.append({
                    "sample_idx": s + 1,
                    "model_raw_output": gen_text,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "latency": round(latency, 3)
                })
            
            # 保存单题结果（后续逻辑不变）
            record = {
                "question_id": idx,
                "question": question,
                "reference_answer": ref_ans,
                "model_outputs": samples,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_latency": round(total_latency, 3),
                "config": {
                    "method_name": method_name,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "use_reflection": use_reflection  # 标记是否启用反思
                }
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nRaw answers saved to → {out_file}")
    return out_file

def evaluate_passk(jsonl_path, pass_k=NUM_SAMPLES):
    """评估Pass@k指标"""
    total, pass_ok = 0, 0
    all_lat, all_pt, all_ct = [], [], []
    
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            ref = rec["reference_answer"]
            # 检查前pass_k个样本是否有正确答案
            if any(grade_answer(extract_answer(o["model_raw_output"]), ref)
                   for o in rec["model_outputs"][:pass_k]):
                pass_ok += 1
            # 收集统计信息
            all_lat.append(rec["total_latency"])
            all_pt.append(rec["total_prompt_tokens"])
            all_ct.append(rec["total_completion_tokens"])
    # 计算汇总指标
    return {
        "total": total,
        "pass@k": round(pass_ok / total, 4),
        "latency_avg": round(sum(all_lat) / len(all_lat), 2),
        "latency_total": round(sum(all_lat), 2),
        "prompt_tokens_total": sum(all_pt),
        "completion_tokens_total": sum(all_ct)
    }