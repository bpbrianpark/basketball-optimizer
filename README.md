# Basketball Shot Form Optimizer

Project Flow:
1. Takes a video of someone shooting a basketball (from a webcam or uploaded file)
2. Uses AI to detect the person's body pose (where their joints are)
3. Calculates angles between body parts (elbow, wrist, shoulder, etc.)
4. Scores how "good" or "bad" the shooting form is
5. Provides feedback on strengths and weaknesses

### How It Works

1. **Video Input**: You record or upload a video of a basketball shot
2. **Pose Detection**: An AI model (YOLOv8) looks at each frame and finds where the person's joints are (shoulders, elbows, wrists, etc.)
3. **Angle Calculation**: We calculate angles between joints (e.g., elbow angle, wrist angle)
4. **Feature Extraction**: We summarize all those angles into key metrics (average elbow angle, consistency, etc.)
5. **Scoring**: A trained machine learning model looks at those metrics and says "this is a good shot" or "this needs work"
6. **Feedback**: The system tells you what you did well and what to improve


## Features
- FastAPI backend for uploading clips and retrieving analysis.
- YOLOv8 pose estimation + OpenCV overlay utilities.
- Pandas/NumPy data processing helpers for feature extraction.
- Scikit-learn training skeleton for classifying good vs. bad form.
- Jupyter notebook template for iterative experiments.

## Prerequisites
- Python 3.11+ (https://www.python.org/downloads/)
- `ffmpeg` (recommended for richer video codec support)

## Setup

### 1. Create and activate a virtual environment

A **virtual environment** is an isolated Python environment that keeps project dependencies separate from other projects. This prevents conflicts between different projects that might need different versions of the same library.

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# On Windows (PowerShell): .venv\Scripts\Activate.ps1
# On Windows (Command Prompt): .venv\Scripts\activate.bat
```

If the above doesn't work, try:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# On Windows (PowerShell): .venv\Scripts\Activate.ps1
# On Windows (Command Prompt): .venv\Scripts\activate.bat
```

**How to know it's working**: Your terminal prompt should show `(.venv)` at the beginning.

**To deactivate** (when you're done working): Type `deactivate`

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all the Python packages needed for the project. This may take a few minutes.

If the above doesn't work, try
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 3. Register the Jupyter kernel

A **Jupyter kernel** is the execution engine that runs code inside Jupyter notebooks. By registering your virtual environment as a kernel, you ensure that notebooks use the correct Python environment and packages.

```bash
python -m ipykernel install --user --name basketball-optimizer --display-name "Basketball Optimizer"
```

If the above isn't visible in your JupyterNotebook, try:
```bash
python3 -m ipykernel install --user --name basketball-optimizer --display-name "Basketball Optimizer"
```

If neither is available, just using the
`.venv (Python 3.10.11)` 
is fine.

**What this does**: Creates a kernel named "Basketball Optimizer" that uses your project's virtual environment.

**How to use it**: 
- In JupyterLab/Notebook: Go to Kernel → Change Kernel → Select "Basketball Optimizer"
- In VS Code: Open Command Palette (Cmd/Ctrl+Shift+P) → "Notebook: Select Notebook Kernel" → Choose "Basketball Optimizer"

### 4. Run the FastAPI backend (dev)

```bash
uvicorn backend.app.main:app --reload
```

The `--reload` flag automatically restarts the server when you make code changes (useful for development).

Once running, you can:
- Visit `http://localhost:8000/docs` for interactive API documentation
- Visit `http://localhost:8000/health` to check if the server is running

## Repository Layout
```
basketball-optimizer/
├── backend/              # FastAPI backend application
│   └── app/
│       ├── api/         # API route handlers
│       ├── services/    # Business logic services
│       └── main.py      # FastAPI entrypoint
├── models/              # ML model utilities and pose pipeline
├── ml/                  # Machine learning training scripts
│   ├── data_pipeline.py
│   └── model_training.py
├── notebooks/           # Jupyter notebooks for analysis
│   └── pose_analysis.ipynb
├── scripts/             # CLI utility scripts
│   └── run_inference.py
├── data/                # Data storage
│   ├── raw/            # Original video files
│   ├── processed/      # Processed features and standardized videos
│   └── labels/         # Label files (CSV with good/bad shot labels)
├── docs/                # Documentation files
├── frontend/            # Frontend code (placeholder for future)
├── config.py           # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Key Files
- `backend/app/main.py` – FastAPI entrypoint and router wiring.
- `backend/app/api/analyze.py` – Request/response schemas and analysis endpoint.
- `backend/app/services/pose_estimator.py` – Pose estimation helpers.
- `ml/data_pipeline.py` – Dataset preprocessing and feature engineering stubs.
- `ml/model_training.py` – Model training workflow scaffold.
- `notebooks/pose_analysis.ipynb` – Analysis notebook template.
- `scripts/run_inference.py` – CLI video analysis stub.
- `config.py` – Centralized configuration (paths, settings, thresholds).

## Troubleshooting

### Virtual Environment Issues
- **Problem**: `python3 -m venv .venv` fails
  - **Solution**: Make sure Python 3.11+ is installed. Check with `python3 --version`

### Package Installation Issues
- **Problem**: `pip install` fails for specific packages
  - **Solution**: Make sure you're in the virtual environment (see `(.venv)` in prompt). Try `pip install --upgrade pip` first.

### Jupyter Kernel Not Showing Up
- **Problem**: "Basketball Optimizer" kernel doesn't appear in Jupyter
  - **Solution**: Make sure you ran the kernel registration command while the virtual environment was activated. Try running it again.

### Import Errors
- **Problem**: `ImportError` when running code
  - **Solution**: Make sure you're using the correct kernel in Jupyter, or that your virtual environment is activated in terminal.
