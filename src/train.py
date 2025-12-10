# src/train.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.config import load_config
from src.data_loader import MovieLensLoader
from src.tmdb_enricher import TMDbEnricher
from src.content import ContentEmbedder
from src.ncf import NeuralCollaborativeFiltering
from dotenv import load_dotenv

def main():
    """Main training pipeline to generate all artifacts."""
    load_dotenv() # Load environment variables from .env file

    cfg = load_config()
    exp_cfg = cfg['experiment']
    paths_cfg = cfg['paths']

    # Dynamically set paths based on dataset
    for key, path in paths_cfg.items():
        paths_cfg[key] = path.format(dataset=exp_cfg['dataset'])

    # --- 1. Load and Enrich Data ---
    print(f"🎬 Loading dataset: {exp_cfg['dataset']}")
    loader = MovieLensLoader(dataset=exp_cfg['dataset'], data_dir=paths_cfg['data_dir'])
    ratings_df, movies_df = loader.load()

    if exp_cfg.get('use_tmdb_enrichment', False):
        api_key = os.getenv('TMDB_API_KEY')
        if api_key:
            print("✨ Enriching movie data with TMDB details...")
            enricher = TMDbEnricher(api_key=api_key)
            movies_df = enricher.enrich(movies_df)
        else:
            print("⚠️ TMDB_API_KEY not found. Skipping enrichment.")

    # --- 2. Create and Save ID Mappings ---
    print("🗺️ Creating and saving ID mappings...")
    user_map = {uid: i for i, uid in enumerate(ratings_df['user_id'].unique())}
    item_map = {mid: i for i, mid in enumerate(movies_df['movie_id'].unique())}

    ratings_df['user_idx'] = ratings_df['user_id'].map(user_map)
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(item_map)
    ratings_df.dropna(subset=['user_idx', 'movie_idx'], inplace=True)
    ratings_df['user_idx'] = ratings_df['user_idx'].astype(int)
    ratings_df['movie_idx'] = ratings_df['movie_idx'].astype(int)

    n_users, n_items = len(user_map), len(item_map)
    print(f"Total unique users: {n_users}, Total unique items: {n_items}")

    # --- 3. Build and Save Content Embeddings ---
    print("🧠 Building content embeddings...")
    embedder = ContentEmbedder(**cfg['content'], cache_dir=paths_cfg['artifacts_dir'])
    embedder.fit_transform(
        movies_df,
        cache_key=os.path.basename(paths_cfg['embeddings_cache']).replace('.npy', '')
    )

    # --- 4. Train NCF Model ---
    print("🚀 Training Neural Collaborative Filtering model...")
    X = ratings_df[['user_idx', 'movie_idx']].values
    y = ratings_df['rating'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=exp_cfg['val_split'], random_state=42
    )

    # ✅ FIX: Separate model params from training params
    ncf_cfg = cfg['ncf'].copy()
    epochs = ncf_cfg.pop('epochs')
    batch_size = ncf_cfg.pop('batch_size')

    ncf = NeuralCollaborativeFiltering(n_users, n_items, **ncf_cfg)
    model = ncf.build()
    ncf.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    # --- 5. Save All Artifacts ---
    print("💾 Saving all artifacts...")
    os.makedirs(paths_cfg['models_dir'], exist_ok=True)
    # Save in the modern .keras format
    model.save(os.path.join(paths_cfg['models_dir'], "ncf_model.keras"))
    movies_df.to_csv(paths_cfg['enriched_data_cache'], index=False)
    
    # Make maps JSON-safe before saving
    safe_user_map = {str(k): int(v) for k, v in user_map.items()}
    safe_item_map = {str(k): int(v) for k, v in item_map.items()}
    with open(paths_cfg['id_maps_path'], 'w') as f:
        json.dump({'user_map': safe_user_map, 'item_map': safe_item_map}, f, indent=4)

    user_history = ratings_df[ratings_df['rating'] >= 4].groupby('user_id')['movie_id'].apply(list).to_dict()
    safe_user_history = {str(k): v for k, v in user_history.items()}
    with open(paths_cfg['user_history_path'], 'w') as f:
        json.dump(safe_user_history, f, indent=4)

    print(f"\n🎉 Training pipeline complete! Artifacts saved in '{paths_cfg['artifacts_dir']}'")

if __name__ == "__main__":
    main()