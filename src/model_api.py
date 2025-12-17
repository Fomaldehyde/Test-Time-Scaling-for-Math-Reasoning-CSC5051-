import os
import json
import time
import requests
from tqdm import tqdm
from retrying import retry
import functools
from .prompts import get_prompt
from config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME

# -------------------- 1. 单次生成 --------------------
@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def generate_answer_api(
        question="",
        prompt="",
        do_sample=False,
        max_new_tokens=1024,
        temperature=0.5
):
    full_prompt = f"{question}\n\n{prompt}".strip()
    messages = [{"role": "user", "content": full_prompt}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature if do_sample else 0.0,
        "max_tokens": max_new_tokens,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{OPENAI_API_BASE}/chat/completions"

    start = time.time()
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    latency = time.time() - start

    result = resp.json()
    text = result["choices"][0]["message"]["content"].strip()
    prompt_tokens = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    return text, prompt_tokens, completion_tokens, latency

# -------------------- 2. 单次反思 --------------------
def reflect_and_correct(question, original_output, temperature=0.3, max_new_tokens=512):
    ref_prompt = get_prompt("reflection").format(
        question=question,
        original_answer=original_output
    )
    corrected, _, _, _ = generate_answer_api(
        question="", prompt=ref_prompt,
        do_sample=False, temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    return corrected

# -------------------- 3. 通用断点续跑装饰器 --------------------
def resume_jsonl(func):
    """通用续跑装饰器：自动跳过已处理行，无需硬编码路径"""
    @functools.wraps(func)
    def wrapper(input_path, output_path, *args, **kwargs):
        # 1. 统计已写行数（仅统计非空行）
        written = 0
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                written = sum(1 for _ in f if _.strip())
        # 2. 统计输入总行数（仅统计非空行）
        total_in = 0
        with open(input_path, encoding="utf-8") as f:
            total_in = sum(1 for _ in f if _.strip())
        # 3. 已完成则跳过
        if written >= total_in:
            print(f"[DONE] {output_path} 已完整（{written}/{total_in}），跳过")
            return
        # 4. 执行函数并传递已写行数
        print(f"[RESUME] 从第{written+1}行开始处理（已写{written}行）")
        func(input_path, output_path, skip_lines=written, *args, **kwargs)
    return wrapper

# -------------------- 4. 底层反射调用 --------------------
@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def _reflect_call(question, original_output, temperature=0.3, max_new=512):
    ref_prompt = get_prompt("reflection").format(
        question=question,
        original_answer=original_output
    )
    ans, pt, ct, _ = generate_answer_api(
        question="", prompt=ref_prompt,
        do_sample=False, temperature=temperature,
        max_new_tokens=max_new
    )
    return ans, pt, ct

# -------------------- 5. 通用反射修正（仅保留反思结果到model_output） --------------------
@resume_jsonl
def refine_generic(
    input_path: str, 
    output_path: str, 
    skip_lines: int, 
    temperature: float = 0.3, 
    max_new_tokens: int = 512,
    sample_idx: int = 0  # 指定处理第几个样本（适配pass@k）
):
    """通用反射修正函数：反射后仅将反思结果存入model_output字段"""
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "a", encoding="utf-8") as fout:
        
        # 跳过已处理行（仅跳过非空行）
        skipped = 0
        while skipped < skip_lines:
            line = fin.readline()
            if not line:
                break
            if line.strip():
                skipped += 1
        
        # 统计剩余行数（用于进度条）
        remaining_lines = sum(1 for _ in fin if _.strip())
        fin.seek(0)  # 重置文件指针
        # 重新跳过已处理行
        skipped = 0
        while skipped < skip_lines:
            line = fin.readline()
            if not line:
                break
            if line.strip():
                skipped += 1
        
        # 处理剩余行
        pbar = tqdm(
            iter(fin.readline, ""),
            desc="Refining (Generic)",
            initial=skip_lines,
            total=skip_lines + remaining_lines
        )
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            
            rec = json.loads(line)
            # 校验样本索引是否越界
            if sample_idx >= len(rec["model_outputs"]):
                print(f"⚠️ 样本索引{sample_idx}越界，跳过question_id={rec.get('question_id', '未知')}")
                continue
            
            # 读取原始model_output（无需兼容其他字段）
            sample = rec["model_outputs"][sample_idx]
            original_ans = sample.get("model_output", "")  # 仅读取原始model_output
            pt1 = sample.get("prompt_tokens", 0)          # 原始生成的token数
            ct1 = sample.get("completion_tokens", 0)
            lat1 = sample.get("latency", 0.0)              # 原始生成的耗时

            # 执行反射修正
            t0 = time.time()
            reflect_ans, pt2, ct2 = _reflect_call(
                rec["question"], original_ans, temperature, max_new_tokens
            )
            lat2 = time.time() - t0  # 反思阶段耗时

            #仅保留反思结果到model_output（覆盖原始值）
            sample.update({
                "model_output": reflect_ans,  # 只存反思后的答案
                "prompt_tokens": pt1 + pt2,   # 累计原始+反思的token数
                "completion_tokens": ct1 + ct2,
                "latency": round(lat1 + lat2, 3)  # 累计原始+反思的耗时
            })
            rec["model_outputs"][sample_idx] = sample

            # 更新全局统计
            rec["total_prompt_tokens"] = rec.get("total_prompt_tokens", 0) + pt2
            rec["total_completion_tokens"] = rec.get("total_completion_tokens", 0) + ct2
            rec["total_latency"] = rec.get("total_latency", 0.0) + round(lat2, 3)

            # 给方法名加后缀（标记已反射）
            if "config" in rec and "method_name" in rec["config"]:
                if not rec["config"]["method_name"].endswith("_refine"):
                    rec["config"]["method_name"] += "_refine"

            # 实时写入修正后的结果
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()