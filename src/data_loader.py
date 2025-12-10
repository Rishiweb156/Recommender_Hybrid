import os
import pandas as pd

class MovieLensLoader:
    """Loads and preprocesses the MovieLens 1M dataset."""
    
    def __init__(self, dataset='ml-1m', data_dir='data'):
        self.dataset = dataset
        # Create path like data/ml-1m
        self.data_dir = os.path.join(data_dir, dataset)
        os.makedirs(self.data_dir, exist_ok=True)

    def load(self):
        """
        Load MovieLens 1M dataset files:
        - ratings.dat (userId::movieId::rating::timestamp)
        - movies.dat  (movieId::title::genres)
        """
        ratings_path = os.path.join(self.data_dir, 'ratings.dat')
        movies_path = os.path.join(self.data_dir, 'movies.dat')

        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            raise FileNotFoundError(
                f"❌ MovieLens 1M dataset not found in {self.data_dir}/\n"
                f"Make sure files 'ratings.dat' and 'movies.dat' exist inside 'data/ml-1m/'."
            )

        print(f"🎬 Loading MovieLens 1M dataset from {self.data_dir} ...")

        # ✅ FIXED: Use ISO-8859-1 encoding to avoid UnicodeDecodeError
        ratings = pd.read_csv(
            ratings_path,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='ISO-8859-1'
        )

        movies = pd.read_csv(
            movies_path,
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='ISO-8859-1'
        )

        print(f"✅ Loaded ratings: {ratings.shape}, movies: {movies.shape}")
        print(f"🧩 Sample movie: {movies.iloc[0]['title']} — Genres: {movies.iloc[0]['genres']}")

        return ratings, movies
