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
        # 1. 统计已写行数
        written = 0
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                written = sum(1 for _ in f)
        # 2. 统计输入总行数
        with open(input_path, encoding="utf-8") as f:
            total_in = sum(1 for _ in f)
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

# -------------------- 5. 断点续跑 --------------------
@resume_jsonl  # 不再硬编码路径，通用装饰器
def refine_generic(  # 改名：通用反射修正
    input_path: str, 
    output_path: str, 
    skip_lines: int, 
    temperature: float = 0.3, 
    max_new_tokens: int = 512,
    sample_idx: int = 0  # 新增：指定处理第几个样本（适配pass@k）
):
    """通用反射修正函数（支持任意生成的jsonl续跑）"""
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "a", encoding="utf-8") as fout:
        
        # 跳过已处理行（续跑核心逻辑，通用）
        for _ in range(skip_lines):
            next(fin)
        
        # 进度条适配（通用）
        remaining_lines = sum(1 for _ in fin)
        fin.seek(0)
        for _ in range(skip_lines):
            next(fin)
        
        # 通用业务逻辑（适配任意jsonl）
        for line in tqdm(fin, desc="Refining (Generic)", initial=skip_lines, total=skip_lines+remaining_lines):
            rec = json.loads(line.strip())
            # 通用：取指定样本（适配pass@k）
            sample = rec["model_outputs"][sample_idx]
            # 通用：兼容round1/round2/model_raw_output字段
            ans1 = sample.get("round1") or sample.get("model_raw_output", "")
            pt1 = sample.get("prompt_tokens", 0)
            ct1 = sample.get("completion_tokens", 0)
            lat1 = sample.get("latency", 0.0)

            # 反射修正（通用逻辑不变）
            t0 = time.time()
            ans2, pt2, ct2 = _reflect_call(rec["question"], ans1, temperature, max_new_tokens)
            lat2 = time.time() - t0

            # 通用：更新样本结构（兼容任意method_name）
            sample.update({
                "round1": ans1,
                "round2": ans2,
                "prompt_tokens": pt1 + pt2,
                "completion_tokens": ct1 + ct2,
                "latency": round(lat1 + lat2, 3)
            })
            rec["model_outputs"][sample_idx] = sample
            # 通用：更新全局统计
            rec["total_prompt_tokens"] = rec.get("total_prompt_tokens", 0) + pt2
            rec["total_completion_tokens"] = rec.get("total_completion_tokens", 0) + ct2
            rec["total_latency"] = rec.get("total_latency", 0.0) + round(lat2, 3)
            # 通用：给方法名加后缀（可选）
            if "config" in rec and "method_name" in rec["config"]:
                if not rec["config"]["method_name"].endswith("_refine"):
                    rec["config"]["method_name"] += "_refine"

            # 续跑核心：实时写入+flush（通用）
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()