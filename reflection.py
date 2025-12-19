import os, json, time
from tqdm import tqdm
from src.model_api import generate_answer_api
from retrying import retry
from config import SAVE_DIR
from src.prompts import get_prompt


@retry(stop_max_attempt_number=3, wait_fixed=1000)
def reflect_and_correct(question, original_output):
    # Use unified reflection prompt from prompts.py for consistency
    check_prompt_template = get_prompt("reflection")
    check_prompt = check_prompt_template.format(
        question=question,
        original_answer=original_output,
    )
    second, pt, ct, _ = generate_answer_api(
        question=question,
        prompt=check_prompt,
        do_sample=False,
        max_new_tokens=4096,
    )
    return second, pt, ct


def add_refine_to_jsonl_local(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- 断点检测 ----
    written = 0
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            written = sum(1 for _ in f)
        print(f"[CHECK] 输出文件已存在，已写 {written} 行")

    # ---- 如果已写完直接退出 ----
    with open(input_path, encoding="utf-8") as f:
        total_in = sum(1 for _ in f)
    if written >= total_in:
        print("[DONE] 文件已完整，无需再跑。")
        return

    # ---- 从断点继续 ----
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "a", encoding="utf-8") as fout:

        # 跳过已写行
        for _ in range(written):
            fin.readline()

        for line in tqdm(fin, desc="Local refining", initial=written, total=total_in):
            rec = json.loads(line)
            r1 = rec["model_outputs"][0]
            ans1 = r1["model_output"]
            pt1, ct1, lat1 = r1["prompt_tokens"], r1["completion_tokens"], r1["latency"]

            t0 = time.time()
            ans2, pt2, ct2 = reflect_and_correct(rec["question"], ans1)
            lat2 = time.time() - t0

            # 更新总计
            rec["model_outputs"] = [{
                "sample_idx": 1,
                "model_output": ans2,
                "prompt_tokens": pt1 + pt2,
                "completion_tokens": ct1 + ct2,
                "latency": round(lat1 + lat2, 3)
            }]
            rec["total_prompt_tokens"] += pt2
            rec["total_completion_tokens"] += ct2
            rec["total_latency"] += round(lat2, 3)
            rec["config"]["method_name"] += "_refine"

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()          # 实时落盘
    print(f"Local reflection finished → {output_path}")


if __name__ == "__main__":
    in_file = "./experiment_results/raw_few_shot_pass@1_seed42.jsonl"
    out_file = "./experiment_results/raw_few_shot_pass@1_seed42_reflection.jsonl"
    add_refine_to_jsonl_local(in_file, out_file)