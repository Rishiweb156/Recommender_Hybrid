# Experiments and Ablations

This notebook-style markdown explains how to run experiments on MovieLens 100K to compare:
- GMF-only
- MLP-only
- NCF (GMF + MLP)
- Content-only (TF-IDF or SBERT)

Steps:
1. Download MovieLens 100K and place in `data/`:
   - `u.data` and `u.item` (from https://grouplens.org/datasets/movielens/100k/)
2. Install requirements: `pip install -r requirements.txt`
3. Run training with flags to toggle GMF/MLP:
   - Hybrid (default): `python -m src.train --dataset ml-100k --use_sbert --epochs 12 --save_dir artifacts`
   - GMF-only: add `--no_mlp`
   - MLP-only: add `--no_gmf`
4. Evaluate ranking metrics produced in training logs (Recall@10, Precision@10, NDCG@10).
5. For ablation, record results in a table:
   - Columns: model, RMSE, Recall@10, NDCG@10

Plotting:
- Plot ranking metrics across models (bar charts)
- Plot training loss and RMSE curves

Notes:
- Use `--use_sbert` to enable SBERT; otherwise TF-IDF-only will be used.
- Ensure artifacts are saved to `artifacts/` and use `app/streamlit_app.py` to visualize results.
