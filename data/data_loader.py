import json
import os
import sys
import random
from datasets import load_dataset

# 获取当前文件（data_loader.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（current_dir的上层目录）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path，这样就能找到config.py
sys.path.append(root_dir)

from config import DATASET_NAME, DATASET_SPLIT, SAMPLE_NUM, RANDOM_SEED


def load_math_dataset(split=DATASET_SPLIT, n=None, save_local=True, local_path=None):
    """
    Load math dataset (MATH-500/GSM8K) with local cache to avoid repeated downloads
    - MATH-500: fields are "problem"/"answer"
    - GSM8K: fields are "question"/"answer"
    """
    # Set default local path (save in data/ folder)
    if local_path is None:
        data_dir = os.path.dirname(__file__)  # Get current file's directory (data/)
        local_path = os.path.join(data_dir, f"{DATASET_NAME.replace('/', '_')}_{split}.json")
    
    # Step 1: Load from local file if exists
    if os.path.exists(local_path):
        print(f"Loading dataset from local file: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            processed_questions = json.load(f)
        # Control test scale
        if n:
            processed_questions = processed_questions[:n]
        print(f"Successfully loaded {len(processed_questions)} questions (local cache)")
        return processed_questions
    
    # Step 2: Download dataset from HuggingFace
    print(f"No local cache found, downloading {DATASET_NAME} ({split})...")
    # Handle different dataset configs (GSM8K needs "main" config)
    if DATASET_NAME == "openai/gsm8k":
        ds = load_dataset(DATASET_NAME, "main", split=split)
    else:
        ds = load_dataset(DATASET_NAME, split=split)

    # Convert to unified {question, answer} format (compatible with both datasets)
    processed_questions = []
    for item in ds:
        # GSM8K uses "question", MATH-500 uses "problem"
        question_key = "question" if DATASET_NAME == "openai/gsm8k" else "problem"
        processed_questions.append({
            "question": item[question_key],  # Unified "question" field
            "answer": item["answer"]         # Unified "answer" field
        })

    # Control test scale
    if n:
        processed_questions = processed_questions[:n]

    # Save to local
    if save_local:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(processed_questions, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to local path: {local_path}")

    print(f"Successfully loaded {len(processed_questions)} questions (first download)")
    return processed_questions


def sample_questions(dataset, num_samples=SAMPLE_NUM, seed=RANDOM_SEED):
    """
    Sample N questions from dataset (fixed seed for reproducibility)
    :param dataset: List of {"question": ..., "answer": ...}
    :param num_samples: Number of questions to sample
    :param seed: Random seed
    :return: Sampled questions list
    """
    random.seed(seed)
    # Ensure num_samples <= dataset length
    num_samples = min(num_samples, len(dataset))
    sampled_questions = random.sample(list(dataset), num_samples)
    print(f"Successfully sampled {len(sampled_questions)} questions for testing")
    return sampled_questions


if __name__ == "__main__":
    # Test dataset loading + sampling (仅用于验证模块功能)
    full_dataset = load_math_dataset(split=DATASET_SPLIT, n=None)
    sampled_questions = sample_questions(full_dataset, num_samples=SAMPLE_NUM)
    # Print first sample to verify
    print("\nFirst sampled question:")
    print(f"Question: {sampled_questions[0]['question'][:100]}...")
    print(f"Answer: {sampled_questions[0]['answer'][:50]}...")