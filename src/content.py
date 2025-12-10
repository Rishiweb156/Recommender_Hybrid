import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False


class ContentEmbedder:
    """Create content embeddings using TF-IDF and optionally Sentence-BERT."""
    def __init__(self, sbert_model='all-MiniLM-L6-v2', max_tfidf_features=2048, use_sbert=True, cache_dir='artifacts'):
        self.use_sbert = use_sbert and _HAS_SBERT
        self.model_name = sbert_model
        self.max_features = max_tfidf_features
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

        if self.use_sbert:
            self.sbert = SentenceTransformer(self.model_name)
        else:
            self.sbert = None

    def fit_transform(self, movies_df, cache_key='content_emb'):
        cache_path = os.path.join(self.cache_dir, f'{cache_key}.npy')
        if os.path.exists(cache_path):
            return np.load(cache_path)

        movies_df = movies_df.fillna('')
        movies_df['text'] = movies_df['title'].astype(str) + ' ' + movies_df.get('genres', '').astype(str)

        if 'overview' in movies_df.columns:
            movies_df['text'] = movies_df['text'] + ' ' + movies_df['overview'].astype(str)

        tfidf_emb = self.vectorizer.fit_transform(movies_df['text']).astype(np.float32)

        if self.sbert:
            sbert_emb = self.sbert.encode(
                movies_df['text'].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True
            ).astype(np.float32)
            combined = np.hstack([tfidf_emb.toarray(), sbert_emb])
        else:
            combined = tfidf_emb.toarray()

        np.save(cache_path, combined)
        return combined
