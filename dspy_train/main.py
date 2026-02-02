import dspy
import mlflow
from .config import Config
from .dataset import load_data
from .modules import CoTModule
from .optimizer import optimize_program

def main():
    # 1. Setup MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    # Enable DSPy Autologging
    # log_compiles: Log the compiled program
    # log_evals: Log evaluation metrics
    mlflow.dspy.autolog(log_compiles=True, log_evals=True)

    # 2. Setup DSPy LM
    lm = dspy.LM(
        model=Config.LM_MODEL_NAME,
        api_base=Config.LM_API_BASE,
        api_key=Config.LM_API_KEY
    )
    dspy.configure(lm=lm)
    
    # 3. Load Data
    print("Loading data...")
    trainset, devset, metric = load_data()
    print(f"Data loaded. Train size: {len(trainset)}, Dev size: {len(devset)}")

    # 4. Initialize Module
    program = CoTModule()
    
    # 5. Run Optimization
    # We can explicitly start a run if we want to add extra tags or params
    with mlflow.start_run(run_name="DSPy_Optimization_Run") as run:
        mlflow.log_param("optimizer", "MIPROv2")
        mlflow.log_param("train_size", len(trainset))
        
        print("Optimizing program...")
        optimized_program = optimize_program(program, trainset, metric, method="mipro")
        
        # 6. Evaluate (Optional, as autolog might capture some, but good to be explicit)
        print("Evaluating optimized program...")
        evaluator = dspy.Evaluate(devset=devset, metric=metric, num_threads=1, display_progress=True)
        score = evaluator(optimized_program)
        print(f"Evaluation Score: {score}")
        
        mlflow.log_metric("final_dev_score", score)
        
        # Save the program locally as well
        optimized_program.save("dspy-train/data/optimized_cot.json")
        mlflow.log_artifact("dspy-train/data/optimized_cot.json")

if __name__ == "__main__":
    main()
