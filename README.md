# DSPy Optimization Pipeline with MLflow

This package (`dspy_train`) acts as a structured framework for optimizing DSPy modules (prompts) using the MIPROv2 optimizer, with full experiment tracking via MLflow.

## Overview

The pipeline allows you to:
1.  Define DSPy Signatures and Modules declaratively.
2.  Optimize the module using the MIPROv2 teleprompter (which selects the best few-shot examples and instructions).
3.  Track every optimization run, including prompt traces and evaluation metrics, in MLflow.
4.  Save the optimized program for production use.

## Structure

```text
dspy_train/
├── config.py       # Configuration (MLflow URI, LLM settings, Hyperparams)
├── dataset.py      # Dataset loading (loading GSM8K for this demo)
├── signatures.py   # DSPy Signatures (Input/Output definitions)
├── modules.py      # DSPy Modules (Chain of Thought logic)
├── optimizer.py    # Optimization logic (MIPROv2 wrapper)
├── main.py         # Entry point to run the pipeline
└── __init__.py     # Package initialization
```

## Setup & Configuration

1.  **Dependencies**: Ensure `dspy` and `mlflow` are installed.
2.  **Configuration**: Edit `dspy_train/config.py` to match your environment:
    *   `MLFLOW_TRACKING_URI`: URL of your MLflow server (default: `http://localhost:5000`)
    *   `LM_API_BASE` / `LM_MODEL_NAME`: Your LLM credentials/endpoint.

## Usage

1.  **Start MLflow** (if not running):
    ```bash
    mlflow ui --port 5000
    ```

2.  **Run the Optimization**:
    Run the package as a module from the project root:
    ```bash
    python -m dspy_train.main
    ```

## MLflow Integration

*   **Autologging**: The pipeline uses `mlflow.dspy.autolog()` to capture traces of calls to the LLM.
*   **Experiments**: Runs are logged under the experiment name defined in `Config`.
*   **Artifacts**: The final optimized program is saved as a JSON artifact (e.g., `optimized_cot.json`).
