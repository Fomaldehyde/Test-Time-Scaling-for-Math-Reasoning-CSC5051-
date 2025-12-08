# 所有Prompt模板（统一维护，便于修改）
PROMPTS = {
    "0_shot": "",
    "base_only_answer": "Give ONLY the final answer WITHOUT explanation, put the answer in \\boxed{}",
    "cot_detailed": "Solve the problem step by step and put the final answer within \\boxed{}.\n",
    "cot_check": "Solve step by step, then double-check your arithmetic then put the corrected final answer in \\boxed{}." ,
    "few_shot": """
Example 1: Q: 3+5=?  Steps: 3+5=8. Answer \\boxed{8}
Example 2: Q: 2x=6?  Steps: x=6/2=3. Answer \\boxed{3}
Now solve the question below step by step and use \\boxed{}.
""",
    "mini_cot": "Think in one sentence, then give the final answer in \\boxed{}.",
    "reflection": """
You just solved the following math problem:
Question: {question}
Your original answer: {original_answer}

Please check your answer for the following issues:
1. Arithmetic errors (addition/subtraction/multiplication/division mistakes)
2. Logical gaps (missing steps, wrong assumptions)
3. Misinterpretation of the question
4. Incorrect unit conversion (if applicable)

If you find errors, correct them and provide the final correct answer in \\boxed{{}}.
If no errors are found, repeat the original answer in \\boxed{{}}.
"""
}

# 对外暴露获取Prompt的函数
def get_prompt(prompt_type: str) -> str:
    if prompt_type not in PROMPTS:
        raise ValueError(f"Prompt type {prompt_type} not found! Available: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_type]