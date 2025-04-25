"""
Configuration settings for the project
"""

# data settings
DATA_PATH = "../../dataset/arxiv_data.csv"
OUTOUT_DIR = "./output"
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42
SAMPLE_RATIO = 0.5 #### edit

# model settings
MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 20
THRESHOLD = 0.5

# confidence settings
CONFIDENCE_METHOD = "avg_max_prob"
CONFIDENCE_THRESHOLD = 0.8 