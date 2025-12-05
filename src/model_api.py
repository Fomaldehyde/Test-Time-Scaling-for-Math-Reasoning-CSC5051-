import time
import requests
from retrying import retry
from config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def generate_answer_api(
    question,
    prompt="",
    do_sample=False,
    max_new_tokens=512,
    temperature=0.7
):
    """调用API生成答案，返回（生成文本、prompt_token数、completion_token数、延迟）"""
    # 构造完整Prompt
    full_prompt = f'{question}\n\n{prompt}'
    # API请求参数
    messages = [{"role": "user", "content": full_prompt}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature if do_sample else 0.0,
        "max_tokens": max_new_tokens,
        "stream": False
    }
    # 请求头
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    # 发送请求并计时
    OPENAI_API_URL = f"{OPENAI_API_BASE}/chat/completions"
    start_time = time.time()
    response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
    response.raise_for_status()  # 抛出HTTP错误
    latency = time.time() - start_time
    # 解析响应
    result = response.json()
    generated_text = result["choices"][0]["message"]["content"].strip()
    prompt_tokens = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    return generated_text, prompt_tokens, completion_tokens, latency


# 反思修正方法
def reflect_and_correct(question, original_output, temperature=0.3, max_new_tokens=1024):
    """
    反思修正：基于初始输出，让模型自查并修正答案
    :param question: 原始问题
    :param original_output: 模型初始生成的输出
    :param temperature: 反思阶段用低温度（避免随机）
    :param max_new_tokens: 修正输出的token上限
    :return: 修正后的输出文本
    """
    from src.prompts import get_prompt
    # 填充反思Prompt
    reflection_prompt = get_prompt("reflection").format(
        question=question,
        original_answer=original_output
    )
    # 调用API生成修正后的答案
    corrected_output, _, _, _ = generate_answer_api(
        question="",  # 反思Prompt已包含问题，这里传空
        prompt=reflection_prompt,
        do_sample=False,  # 反思阶段不采样，保证稳定性
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    return corrected_output