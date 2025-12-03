#!/usr/bin/env python3
"""
Creates a clean, interview-friendly project structure for:
- Task 1: 3D CNN + OpenPose skeleton pipeline
- Task 2: simple web app using the trained model as inference engine

Usage:
  python create_project_structure.py /path/to/new_project
  # or, from the desired directory:
  python create_project_structure.py .
"""

from __future__ import annotations
import argparse
from pathlib import Path
import textwrap

DEFAULT_FILES = {
    "README.md": """# Fall Recognition Prototype (Take‑Home)

## Quick start
1) Create env + install deps
2) Download/prepare data (KTH) into `data/raw/`
3) Extract skeletons (OpenPose) into `data/processed/`
4) Train: `python -m src.modeling.train`
5) Evaluate: `python -m src.modeling.evaluate`
6) Web demo: `streamlit run src/app/streamlit_app.py`
""",
    "requirements.txt": """# Core
numpy
pandas
opencv-python
scikit-learn
matplotlib

# Deep learning (choose one stack)
torch
torchvision

# Web demo
streamlit

# Optional (only if you actually use them)
# fastapi
# uvicorn
# python-multipart
""",
    ".gitignore": """__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
data/raw/
data/processed/
outputs/
checkpoints/
""",
    "src/__init__.py": "",
    "src/config.py": """\"\"\"Central configuration (paths, constants).\"\"\"\n\nfrom pathlib import Path\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nDATA_DIR = PROJECT_ROOT / 'data'\nRAW_DIR = DATA_DIR / 'raw'\nPROCESSED_DIR = DATA_DIR / 'processed'\nOUTPUTS_DIR = PROJECT_ROOT / 'outputs'\nCHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'\n\n# Example defaults\nSEED = 42\nNUM_WORKERS = 4\n""",
    "src/data/__init__.py": "",
    "src/data/preprocessing.py": """\"\"\"Video decoding + sampling into fixed-length clips.\"\"\"\n\n# TODO: implement\n""",
    "src/data/openpose_extract.py": """\"\"\"Run OpenPose and store per-frame skeleton keypoints.\"\"\"\n\n# TODO: implement\n""",
    "src/data/dataset.py": """\"\"\"PyTorch Dataset that loads skeleton tensors + labels.\"\"\"\n\n# TODO: implement\n""",
    "src/modeling/__init__.py": "",
    "src/modeling/model_3dcnn.py": """\"\"\"3D CNN definition for skeleton/action recognition.\"\"\"\n\n# TODO: implement\n""",
    "src/modeling/train.py": """\"\"\"Training entrypoint: trains model and saves checkpoints + accuracy curve.\"\"\"\n\ndef main():\n    raise SystemExit('TODO: implement training loop')\n\nif __name__ == '__main__':\n    main()\n""",
    "src/modeling/evaluate.py": """\"\"\"Evaluation entrypoint: confusion matrix + metrics on test set.\"\"\"\n\ndef main():\n    raise SystemExit('TODO: implement evaluation')\n\nif __name__ == '__main__':\n    main()\n""",
    "src/modeling/metrics.py": """\"\"\"Metrics helpers (confusion matrix, accuracy).\"\"\"\n\n# TODO: implement\n""",
    "src/app/__init__.py": "",
    "src/app/streamlit_app.py": """\"\"\"Streamlit demo: upload video -> run inference -> show class probabilities.\"\"\"\n\nimport streamlit as st\n\nst.set_page_config(page_title='Fall Recognition Demo', layout='centered')\n\nst.title('Fall Recognition Demo')\n\nuploaded = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov', 'mkv'])\n\nif uploaded is not None:\n    st.video(uploaded)\n\n    if st.button('Run prediction'):\n        # TODO: load model, run preprocessing + (optional) OpenPose step, return probabilities\n        st.info('TODO: implement inference. Show probabilities here.')\n""",
    "docs/report.md": """# Report Draft Notes (max 5 pages in final Word/PDF)\n\n- Task 1 (≤3 pages):\n  - Architecture diagram\n  - Code snippet screenshots\n  - Pipeline description\n  - Results: accuracy vs epoch + confusion matrix\n  - Objective 3: methodology for elderly falls (no code)\n\n- Task 2 (≤2 pages):\n  - Web app screenshots\n  - Narrative + code snippets\n""",
}

DIRS = [
    "data/raw",
    "data/processed",
    "data/external",
    "checkpoints",
    "outputs/figures",
    "outputs/logs",
    "docs",
    "src/data",
    "src/modeling",
    "src/app",
    "notebooks",
    "scripts",
]

def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create project scaffold for the fall-recognition take-home.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python create_project_structure.py my_takehome_project
        """),
    )
    parser.add_argument("root", nargs="?", default=".", help="Project root directory to create/use.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    for rel, content in DEFAULT_FILES.items():
        write_file(root / rel, content)

    print(f"✅ Project scaffold created at: {root}")
    print("Next steps:")
    print("  1) Put KTH videos in data/raw/")
    print("  2) Implement OpenPose extraction + dataset + model")
    print("  3) Train/evaluate and capture screenshots/plots for the 5-page report")
    print("  4) Run demo: streamlit run src/app/streamlit_app.py")

if __name__ == "__main__":
    main()
