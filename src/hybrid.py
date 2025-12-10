import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class HybridSystem:
    """Combines NCF and Content-Based scores for two-stage recommendations."""
    def __init__(self, ncf_model, content_embeddings, movies_index_map, ann_index, alpha=0.7):
        self.ncf = ncf_model
        self.content_embeddings = content_embeddings.astype(np.float32)
        self.movie_ids = np.array(movies_index_map)
        self.id_to_index = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.ann = ann_index
        self.alpha = alpha
        self.user_history = defaultdict(list)

    def fit_user_history(self, ratings_df, min_rating=4):
        """Builds a map of users to the movies they liked."""
        positive_ratings = ratings_df[ratings_df['rating'] >= min_rating]
        for user, group in positive_ratings.groupby('user_id'):
            self.user_history[user] = group['movie_id'].tolist()

    def _generate_candidates(self, user_id, top_k=500):
        """Stage 1: Retrieve candidate items using content similarity."""
        liked_movies = self.user_history.get(user_id, [])
        if not liked_movies:
            # Cold start: return top popular items (here, just the first k as a proxy)
            return np.arange(min(top_k, len(self.movie_ids)))

        liked_indices = [self.id_to_index[m] for m in liked_movies if m in self.id_to_index]
        if not liked_indices:
            return np.arange(min(top_k, len(self.movie_ids)))

        # Create a user profile vector by averaging liked item embeddings
        user_profile_vector = np.mean(self.content_embeddings[liked_indices], axis=0, keepdims=True)
        
        indices, _ = self.ann.query(user_profile_vector, top_k=top_k)
        return indices.flatten()

    def recommend(self, user_id, n_recommendations=10, return_scores=True):
        """Stage 2: Rank candidate items using a hybrid score."""
        candidate_indices = self._generate_candidates(user_id)
        candidate_movie_ids = self.movie_ids[candidate_indices]

        # Get NCF scores (collaborative signal)
        # The ncf model wrapper is expected to handle ID mapping
        ncf_scores = self.ncf.predict_batch(np.full(len(candidate_movie_ids), user_id), candidate_movie_ids)
        
        # Get content scores (content signal)
        liked_movies = self.user_history.get(user_id, [])
        content_scores = np.zeros(len(candidate_indices))
        if liked_movies:
            liked_indices = [self.id_to_index[m] for m in liked_movies if m in self.id_to_index]
            if liked_indices:
                liked_embeddings = self.content_embeddings[liked_indices]
                candidate_embeddings = self.content_embeddings[candidate_indices]
                # Calculate mean similarity between each candidate and all liked items
                sim_matrix = cosine_similarity(candidate_embeddings, liked_embeddings)
                content_scores = np.mean(sim_matrix, axis=1)

        # Normalize scores to be in [0, 1] range for stable blending
        if ncf_scores.max() > ncf_scores.min():
            ncf_scores = (ncf_scores - ncf_scores.min()) / (ncf_scores.max() - ncf_scores.min())
        if content_scores.max() > content_scores.min():
            content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
            
        # Blend scores
        hybrid_scores = self.alpha * ncf_scores + (1 - self.alpha) * content_scores
        
        # Get top N recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        
        results = [(
            int(candidate_movie_ids[i]),
            float(hybrid_scores[i]),
            float(ncf_scores[i]),
            float(content_scores[i])
        ) for i in top_indices]
        
        return results if return_scores else [r[0] for r in results]