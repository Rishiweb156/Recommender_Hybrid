# src/tmdb_enricher.py
import os
import pandas as pd
import requests
from tqdm import tqdm
from typing import Tuple, Optional

class TMDbEnricher:
    """Fetches movie overviews and poster URLs from The Movie Database (TMDb) API."""
    def __init__(self, api_key: str, delay: float = 0.1):
        if not api_key:
            raise ValueError("TMDb API key is required.")
        self.api_key = api_key
        self.delay = delay
        self.session = requests.Session() # Use a session for connection pooling

    def _get_movie_details(self, title: str, year: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
        """Fetches details for a single movie."""
        base_url = "https://api.themoviedb.org/3/search/movie"
        params = {'api_key': self.api_key, 'query': title}
        if year:
            params['year'] = year
        
        try:
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            results = response.json().get('results', [])
            if results:
                movie = results[0]
                overview = movie.get('overview', '')
                poster = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie.get('poster_path') else None
                return overview, poster
        except requests.RequestException as e:
            print(f"API request failed for title '{title}': {e}")
        return None, None

    def enrich(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Enriches a DataFrame of movies with TMDb metadata."""
        tqdm.pandas(desc="Enriching movie data via TMDb")
        
        # Extract year from title for better matching, e.g., "Toy Story (1995)"
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float).fillna(0).astype(int)
        movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()
        
        details = movies_df.progress_apply(
            lambda row: self._get_movie_details(row['title_clean'], row['year'] if row['year'] > 0 else None),
            axis=1, result_type='expand'
        )
        movies_df[['overview', 'poster_path']] = details
        return movies_df.drop(columns=['year', 'title_clean'])