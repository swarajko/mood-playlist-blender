import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
    def process_and_save_tracks(self, tracks, audio_features, user_id):
        """Combine track info with audio features and save to CSV"""
        # Create dataframes
        tracks_df = pd.DataFrame(tracks)
        features_df = pd.DataFrame(audio_features)

        # If we couldn't fetch any audio features, abort gracefully
        if features_df.empty:
            return None
        
        # Normalize column names so both sides have 'id'
        if 'track_id' in features_df.columns and 'id' not in features_df.columns:
            features_df = features_df.rename(columns={'track_id': 'id'})

        if 'track_id' in tracks_df.columns and 'id' not in tracks_df.columns:
            tracks_df = tracks_df.rename(columns={'track_id': 'id'})

        # Drop rows without IDs (local files, podcasts, etc.)
        tracks_df = tracks_df[tracks_df['id'].notna()]
        # Ensure 'id' exists in features_df
        if 'id' not in features_df.columns:
            if 'track_id' in features_df.columns:
                features_df = features_df.rename(columns={'track_id': 'id'})
            else:
                # Sometimes Spotify audio features are missing 'id',
                # but 'uri' has it (like 'spotify:track:xxxxx')
                if 'uri' in features_df.columns:
                    features_df['id'] = features_df['uri'].str.split(':').str[-1]
                else:
                    raise KeyError("features_df is missing both 'id' and 'track_id'. Got columns: "+ str(features_df.columns))
        # Now drop rows without ids
        features_df = features_df[features_df['id'].notna()]


        # Merge safely (avoid duplicate columns from both frames)
        df = tracks_df.merge(features_df, on='id', how='inner', suffixes=('', '_feat'))

        
        # Add user_id
        df['user_id'] = user_id
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        filename = f'data/user_{user_id}_tracks.csv'
        df.to_csv(filename, index=False)
        
        return df
    
    def load_user_data(self, user_id):
        """Load user's track data from CSV"""
        filename = f'data/user_{user_id}_tracks.csv'
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return None
    
    def normalize_features(self, df):
        """Normalize audio features for model input"""
        # Normalize loudness to 0-1 range (it's typically -60 to 0)
        df['loudness'] = (df['loudness'] + 60) / 60
        
        # Normalize tempo to 0-1 range (typically 0-250 BPM)
        df['tempo'] = df['tempo'] / 250
        
        # All other features are already 0-1
        features = df[self.feature_columns].values
        
        return features
    
    def get_mood_vector(self, mood_name):
        """Convert mood name to feature vector (predefined mood profiles)"""
        mood_profiles = {
            'happy': [0.7, 0.8, 0.7, 0.1, 0.2, 0.1, 0.3, 0.9, 0.7],
            'sad': [0.3, 0.2, 0.3, 0.1, 0.7, 0.2, 0.1, 0.1, 0.3],
            'energetic': [0.9, 0.95, 0.8, 0.2, 0.1, 0.1, 0.4, 0.7, 0.85],
            'calm': [0.3, 0.2, 0.2, 0.05, 0.8, 0.4, 0.1, 0.5, 0.3],
            'angry': [0.6, 0.9, 0.9, 0.3, 0.1, 0.2, 0.2, 0.2, 0.8],
            'romantic': [0.5, 0.4, 0.3, 0.1, 0.6, 0.3, 0.1, 0.7, 0.4],
            'melancholic': [0.3, 0.3, 0.3, 0.1, 0.6, 0.3, 0.1, 0.2, 0.3],
            'party': [0.95, 0.9, 0.85, 0.2, 0.1, 0.1, 0.5, 0.85, 0.8],
            'focus': [0.3, 0.5, 0.4, 0.05, 0.5, 0.7, 0.05, 0.4, 0.5],
            'chill': [0.5, 0.3, 0.3, 0.1, 0.6, 0.3, 0.1, 0.6, 0.4]
        }
        
        return np.array(mood_profiles.get(mood_name.lower(), mood_profiles['happy']))
    
    def interpolate_moods(self, mood_a_vec, mood_b_vec, n_songs, duration_split):
        """Create interpolation path between two moods"""
        # Determine split ratios
        if duration_split == 'short':
            split_a, split_b = 3, 7
        elif duration_split == 'long':
            split_a, split_b = 7, 3
        else:  # equal
            split_a, split_b = 5, 5
        
        # Create weighted interpolation
        interpolated = []
        for i in range(n_songs):
            if i < split_a:
                # Closer to mood A
                weight = 1 - (i / split_a) * 0.5
            else:
                # Closer to mood B
                idx_in_b = i - split_a
                weight = 0.5 - (idx_in_b / split_b) * 0.5
            
            vec = mood_a_vec * weight + mood_b_vec * (1 - weight)
            interpolated.append(vec)
        
        return np.array(interpolated)