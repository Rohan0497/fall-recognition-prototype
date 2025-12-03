# Fall Recognition Prototype 

## Quick start
1) Create env + install deps
2) Download/prepare data (KTH) into `data/raw/`
3) Extract skeletons (OpenPose) into `data/processed/`
4) Train: `python -m src.modeling.train`
5) Evaluate: `python -m src.modeling.evaluate`
6) Web demo: `streamlit run src/app/streamlit_app.py`
