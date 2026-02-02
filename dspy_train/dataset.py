import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from .config import Config

def load_data():
    """
    Loads the GSM8K dataset and splits it into training and dev sets
    based on the configuration.
    """
    gsm8k = GSM8K()
    
    # Using a subset for faster demonstration
    trainset = gsm8k.train[:Config.TRAIN_SIZE]
    devset = gsm8k.dev[:Config.DEV_SIZE]
    
    return trainset, devset, gsm8k_metric
