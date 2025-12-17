import os
import json
import time
from tqdm import tqdm
from src.model_api import generate_answer_api, reflect_and_correct
from config import SAVE_DIR, NUM_SAMPLES, RANDOM_SEED

def generate_and_save_answers(
    questions_to_test,
    method_name="default",
    prompt="",
    num_samples=NUM_SAMPLES,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024,
    use_reflection=False
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_file = f"{SAVE_DIR}/raw_{method_name}_pass@{num_samples}_seed{RANDOM_SEED}.jsonl"

    # 断点续跑
    generated_ids = set()
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    generated_ids.add(rec["question_id"])
        if len(generated_ids) > 0:
            print(f"[续跑] {out_file} 已生成 {len(generated_ids)} 题，跳过这些题继续...")

    with open(out_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(total=len(questions_to_test),
                    initial=len(generated_ids),
                    desc=f"Generating {method_name}")
        
        for idx, item in enumerate(questions_to_test, 1):
            if idx in generated_ids:
                pbar.update(1)
                continue
            
            question = item["question"]
            ref_ans = item["answer"]
            samples = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_latency = 0.0

            for s in range(num_samples):
                t0 = time.time()
                ans1, pt1, ct1, _ = generate_answer_api(
                    question=question,
                    prompt=prompt,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens
                )

                if use_reflection:
                    ans2, pt2, ct2, _ = reflect_and_correct(
                        question=question,
                        original_output=ans1
                    )
                else:
                    ans2, pt2, ct2 = ans1, 0, 0

                latency = time.time() - t0
                total_prompt_tokens += pt1 + pt2
                total_completion_tokens += ct1 + ct2
                total_latency += latency

                samples.append({
                    "sample_idx": s + 1,
                    "model_output": ans2,
                    "prompt_tokens": pt1 + pt2,
                    "completion_tokens": ct1 + ct2,
                    "latency": round(latency, 3)
                })

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
                    "use_reflection": use_reflection
                }
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            pbar.update(1)
    pbar.close()
    return out_file