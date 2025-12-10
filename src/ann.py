import numpy as np
from typing import Tuple

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors

class AnnIndex:
    """A wrapper for Approximate Nearest Neighbor search using FAISS or scikit-learn."""
    def __init__(self, embeddings: np.ndarray, index_type: str = 'auto', metric: str = 'cosine'):
        self.embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        self.metric = metric
        self.index_type = index_type if index_type != 'auto' else ('faiss' if _HAS_FAISS else 'sklearn')
        self._index = None
        self._build_index()

    def _build_index(self):
        """Builds the ANN index based on the selected backend."""
        if self.index_type == 'faiss' and _HAS_FAISS:
            d = self.embeddings.shape[1]
            if self.metric == 'cosine':
                faiss.normalize_L2(self.embeddings)
                self._index = faiss.IndexFlatIP(d)
                self._index.add(self.embeddings)
            else: # euclidean
                self._index = faiss.IndexFlatL2(d)
                self._index.add(self.embeddings)
        else:
            self._index = NearestNeighbors(n_neighbors=100, metric=self.metric, algorithm='brute')
            self._index.fit(self.embeddings)

    def query(self, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Queries the index to find the top_k nearest neighbors."""
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        
        if self.index_type == 'faiss' and _HAS_FAISS:
            if self.metric == 'cosine':
                faiss.normalize_L2(vectors)
            distances, indices = self._index.search(vectors, top_k)
            # FAISS IP returns similarities, convert to distances for consistency
            if self.metric == 'cosine':
                distances = 1 - distances
            return indices, distances
        else:
            distances, indices = self._index.kneighbors(vectors, n_neighbors=top_k)
            return indices, distances