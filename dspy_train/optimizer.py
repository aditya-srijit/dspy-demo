import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShotWithRandomSearch
import mlflow
from .config import Config

def optimize_program(program, trainset, metric, method="mipro"):
    """
    Optimizes the DSPy program using the specified method.
    Wraps the compilation in an MLflow run.
    """
    print(f"Starting optimization using {method}...")
    
    if method == "mipro":
        teleprompter = MIPROv2(metric=metric, auto="light")
        
        # MIPROv2 optimization
        optimized_program = teleprompter.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=Config.MAX_BOOTSTRAPPED_DEMOS,
            max_labeled_demos=Config.MAX_LABELED_DEMOS,
            requires_permission_to_run=False,
        )
    elif method == "bootstrap":
        teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=Config.MAX_BOOTSTRAPPED_DEMOS)
        optimized_program = teleprompter.compile(program, trainset=trainset)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    print("Optimization complete.")
    return optimized_program
