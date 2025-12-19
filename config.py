import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
#MODEL_NAME = "Qwen/Qwen3-8B"
#MODEL_NAME = "THUDM/GLM-Z1-9B-0414"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"  

SAVE_DIR = "./experiment_results"  # 实验结果保存目录
NUM_SAMPLES = 1                    # 默认生成样本数量（k）
MAX_NEW_TOKENS_BASE = 1024         # 基础方法的token上限
MAX_NEW_TOKENS_LONG = 4096         # 反思方法的token上限
TEMPERATURE_BASE = 0.7             # 基础方法温度

# ==================== 数据集配置 ====================
DATASET_NAME = "HuggingFaceH4/MATH-500"      # 数据集名称
#DATASET_NAME = "math-ai/aime25"      
DATASET_SPLIT = "test"             # 数据集拆分
SAMPLE_NUM = 50                    # 测试题数量
RANDOM_SEED = 42                  # 随机种子