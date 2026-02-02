import os

class Config:
    # MLflow Settings
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME = "DSPy_Optimization_Experiment"
    
    # LLM Settings
    LM_MODEL_NAME = "openai/qwen/qwen3-32b"
    LM_API_BASE = "http://192.168.1.55:1234/v1/"
    LM_API_KEY = "local"
    
    # Dataset Settings
    TRAIN_SIZE = 10  # Keeping it small for demo purposes
    DEV_SIZE = 10
    
    # Optimization Settings
    MAX_BOOTSTRAPPED_DEMOS = 3
    MAX_LABELED_DEMOS = 4
