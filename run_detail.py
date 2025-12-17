import os
import sys
import json
import glob
import re
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from config import NUM_SAMPLES
from src.grade import grade_answer, extract_answer

TARGET_DIR = "./experiment_results"
DETAIL_DIR = os.path.join(TARGET_DIR, "detailed")

def export_detail_for_file(jsonl_path, pass_k=NUM_SAMPLES, detail_dir=DETAIL_DIR):
    os.makedirs(detail_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(jsonl_path))[0]
    # 去掉 raw_ 前缀
    if base.startswith("raw_"):
        base = base[4:]
    out_path = os.path.join(detail_dir, f"{base}_detailed.jsonl")

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ref = rec.get("reference_answer", "")
            outputs = rec.get("model_outputs", [])[:pass_k]

            answers = []
            for o in outputs:
                raw_ans = o.get("round2") or o.get("model_output", "")
                answers.append(extract_answer(raw_ans))
            voted_ans = Counter(answers).most_common(1)[0][0] if answers else ""

            records.append({
                "question_id": rec.get("question_id"),
                "reference_answer": ref,
                "sample_answers": answers,
                "model_answer": voted_ans,
                "is_correct": bool(grade_answer(voted_ans, ref))
            })

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DETAIL] Saved: {out_path}")
    return out_path


def export_all_details(target_dir=TARGET_DIR, detail_dir=DETAIL_DIR, default_k=NUM_SAMPLES):
    jsonl_files = glob.glob(os.path.join(target_dir, "**/raw*.jsonl"), recursive=True)
    if not jsonl_files:
        print(f"[DETAIL] No raw*.jsonl found in {target_dir}")
        return

    print(f"[DETAIL] Found {len(jsonl_files)} files. Exporting details → {detail_dir}")
    for fp in jsonl_files:
        base = os.path.basename(fp)
        k_match = re.search(r"pass@(\d+)", base)
        file_k = int(k_match.group(1)) if k_match else default_k
        export_detail_for_file(fp, pass_k=file_k, detail_dir=detail_dir)


def main():
    export_all_details()


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
