import json
from typing import Counter
from config import NUM_SAMPLES
from src.grade import grade_answer, extract_answer

def evaluate_passk(jsonl_path, pass_k=NUM_SAMPLES):
    """兼容 refine 格式：优先 round2，否则回退 model_raw_output"""
    total, pass_ok = 0, 0
    all_lat, all_pt, all_ct = [], [], []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            ref = rec["reference_answer"]

            # 投票：优先用 round2，没有就回退旧字段
            if any(grade_answer(extract_answer(
                    o.get("round2") or o["model_raw_output"]), ref)
                    for o in rec["model_outputs"][:pass_k]):
                pass_ok += 1

            all_lat.append(rec["total_latency"])
            all_pt.append(rec["total_prompt_tokens"])
            all_ct.append(rec["total_completion_tokens"])

    return {
        "total": total,
        "pass@k": round(pass_ok / total, 4),
        "latency_avg": round(sum(all_lat) / len(all_lat), 2),
        "latency_total": round(sum(all_lat), 2),
        "prompt_tokens_total": sum(all_pt),
        "completion_tokens_total": sum(all_ct)
    }


def self_consistency_passk(jsonl_path, pass_k=NUM_SAMPLES):
    """
    自一致性的评估
    核心逻辑：
    1. 对每道题的k条路径，提取答案（优先round2）并做多数投票
    2. 仅用投票选出的答案和标准答案比对
    3. 返回格式和原evaluate_passk完全一致
    """
    total, sc_correct = 0, 0  # total=总题数，sc_correct=自一致性投票后正确数
    all_lat, all_pt, all_ct = [], [], []  # 资源消耗统计

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            ref = rec["reference_answer"]
            outputs = rec["model_outputs"][:pass_k]  # 取前k条路径

            # ========== 核心：提取k条路径的答案 + 多数投票 ==========
            # 1. 提取每条路径的答案（优先round2，和原逻辑一致）
            path_answers = []
            for o in outputs:
                raw_ans = o.get("round2") or o["model_raw_output"]  # 兼容round1/round2
                norm_ans = extract_answer(raw_ans)
                path_answers.append(norm_ans)
            
            # 2. 多数投票（盲投票：仅统计答案出现次数，无标准答案参与）
            if path_answers:
                vote_counter = Counter(path_answers)
                # 选出现次数最多的答案（平局时选第一条）
                voted_ans = vote_counter.most_common(1)[0][0]
            else:
                voted_ans = ""

            # ========== 判断投票答案是否正确 ==========
            if grade_answer(voted_ans, ref):
                sc_correct += 1

            # ========== 资源消耗统计（和原逻辑一致） ==========
            all_lat.append(rec["total_latency"])
            all_pt.append(rec["total_prompt_tokens"])
            all_ct.append(rec["total_completion_tokens"])

    # 返回格式和原evaluate_passk完全一致
    return {
        "total": total,
        "pass@k": round(sc_correct / total, 4),  # 这里实际是自一致性准确率
        "latency_avg": round(sum(all_lat) / len(all_lat), 2),
        "latency_total": round(sum(all_lat), 2),
        "prompt_tokens_total": sum(all_pt),
        "completion_tokens_total": sum(all_ct)
    }
