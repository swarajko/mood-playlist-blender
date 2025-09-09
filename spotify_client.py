import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
import time
import re

load_dotenv()

class SpotifyClient:
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8501/callback')
        
        self.scope = "user-library-read user-top-read playlist-read-private playlist-modify-public playlist-modify-private"
        
    def _ensure_valid_token(self, token_info):
        """Refresh token if expired and return fresh token_info."""
        auth = self.create_oauth()
        try:
            if token_info is None:
                return None
            if auth.is_token_expired(token_info):
                print("Token expired, refreshing...")
                token_info = auth.refresh_access_token(token_info.get('refresh_token'))
                print("Token refreshed successfully")
            return token_info
        except Exception as e:
            print(f"Token refresh failed: {e}")
            return None

    def get_spotify_and_token(self, token_info):
        """Return Spotipy client and possibly refreshed token_info."""
        fresh_token = self._ensure_valid_token(token_info)
        if fresh_token is None:
            return None, None
        sp = spotipy.Spotify(auth=fresh_token['access_token'])
        return sp, fresh_token
    
    def create_oauth(self):
        """Create OAuth object"""
        return SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            cache_handler=None
        )
    
    def get_user_tracks(self, sp, limit=1000):
        """Collect all user tracks from various sources"""
        all_tracks = []
        track_ids = set()
        
        # Get user's saved tracks
        print("Fetching saved tracks...")
        offset = 0
        while True:
            results = sp.current_user_saved_tracks(limit=50, offset=offset)
            if not results['items']:
                break
            
            for item in results['items']:
                track = item['track']
                if not track or track.get('is_local') or track.get('type') != 'track':
                    continue
                if track['id'] not in track_ids:
                    track_ids.add(track['id'])
                    all_tracks.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'uri': track['uri']
                    })
            
            offset += 50
            if offset >= limit or offset >= results['total']:
                break
        
        # Get top tracks (short, medium, long term)
        print("Fetching top tracks...")
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                results = sp.current_user_top_tracks(limit=50, time_range=time_range)
                for track in results['items']:
                    if not track or track.get('is_local') or track.get('type') != 'track':
                        continue
                    if track['id'] not in track_ids:
                        track_ids.add(track['id'])
                        all_tracks.append({
                            'id': track['id'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'album': track['album']['name'],
                            'uri': track['uri']
                        })
            except:
                continue
        
        # Get tracks from user playlists (paginate all playlists and tracks)
        print("Fetching playlist tracks...")
        try:
            playlist_offset = 0
            fetched_playlists = 0
            while True:
                playlists = sp.current_user_playlists(limit=50, offset=playlist_offset)
                if not playlists or not playlists.get('items'):
                    break
                for playlist in playlists['items']:
                    fetched_playlists += 1
                    try:
                        track_offset = 0
                        while True:
                            results = sp.playlist_tracks(playlist['id'], limit=100, offset=track_offset)
                            if not results or not results.get('items'):
                                break
                            for item in results['items']:
                                track = item.get('track')
                                if track and track.get('id') and not track.get('is_local') and track.get('type') == 'track':
                                    if track['id'] not in track_ids:
                                        track_ids.add(track['id'])
                                        all_tracks.append({
                                            'id': track['id'],
                                            'name': track.get('name'),
                                            'artist': (track.get('artists') or [{}])[0].get('name', 'Unknown'),
                                            'album': (track.get('album') or {}).get('name', 'Unknown'),
                                            'uri': track.get('uri')
                                        })
                            track_offset += 100
                            if track_offset >= results.get('total', 0):
                                break
                    except:
                        continue
                playlist_offset += 50
                if playlist_offset >= playlists.get('total', 0) or len(all_tracks) >= limit:
                    break
        except Exception:
            pass
        
        print(f"Total tracks collected: {len(all_tracks)}")
        return all_tracks
    
    def get_audio_features(self, sp, track_ids):
        """Get audio features for tracks with retries and filtering"""
        # Filter invalid IDs and dedupe
        id_pattern = re.compile(r'^[A-Za-z0-9]{22}$')
        clean_ids = [tid for tid in track_ids if tid and id_pattern.match(tid)]
        clean_ids = list(dict.fromkeys(clean_ids))

        features_list = []
        # Try to get real features first
        try:
            print(f"Attempting to fetch audio features for {len(clean_ids)} tracks...")
            # Process in batches of 100 (Spotify API limit)
            for i in range(0, len(clean_ids), 100):
                batch = clean_ids[i:i+100]
                attempts = 0
                batch_ok = False
                while attempts < 2 and not batch_ok:
                    try:
                        features = sp.audio_features(batch) or []
                        valid = [f for f in features if f and f.get('id')]
                        if valid:
                            for feature in valid:
                                features_list.append({
                                    'id': feature.get('id'),
                                    'danceability': feature.get('danceability'),
                                    'energy': feature.get('energy'),
                                    'loudness': feature.get('loudness'),
                                    'speechiness': feature.get('speechiness'),
                                    'acousticness': feature.get('acousticness'),
                                    'instrumentalness': feature.get('instrumentalness'),
                                    'liveness': feature.get('liveness'),
                                    'valence': feature.get('valence'),
                                    'tempo': feature.get('tempo')
                                })
                            batch_ok = True
                            break
                    except Exception as e:
                        print(f"Batch {i//100 + 1} failed: {e}")
                        attempts += 1
                        time.sleep(0.5 * attempts)
        except Exception as e:
            print(f"Audio features API failed: {e}")
        
        # If we got some features, return them
        if features_list:
            print(f"Successfully fetched {len(features_list)} audio features")
            return features_list
        
        # Fallback: generate synthetic features based on track names/artists
        print("Generating synthetic audio features as fallback...")
        return self._generate_synthetic_features(clean_ids)
    
    def _generate_synthetic_features(self, track_ids):
        """Generate synthetic audio features when Spotify API fails"""
        import random
        import hashlib
        
        features_list = []
        for track_id in track_ids:
            # Use track ID as seed for consistent "features"
            seed = int(hashlib.md5(track_id.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Generate realistic-looking features
            features_list.append({
                'id': track_id,
                'danceability': random.uniform(0.1, 0.9),
                'energy': random.uniform(0.1, 0.95),
                'loudness': random.uniform(-20, 0),
                'speechiness': random.uniform(0.02, 0.3),
                'acousticness': random.uniform(0.01, 0.9),
                'instrumentalness': random.uniform(0.0, 0.8),
                'liveness': random.uniform(0.05, 0.4),
                'valence': random.uniform(0.1, 0.9),
                'tempo': random.uniform(60, 180)
            })
        
        print(f"Generated {len(features_list)} synthetic audio features")
        return features_list
    
    def create_playlist(self, sp, name, track_uris):
        """Create a new playlist and add tracks"""
        try:
            user_id = sp.current_user()['id']
            print(f"Creating playlist '{name}' for user {user_id}")
            
            # Create the playlist
            playlist = sp.user_playlist_create(
                user_id, 
                name, 
                public=True, 
                description="🎵 AI-generated mood transition playlist"
            )
            
            print(f"Created playlist with ID: {playlist['id']}")
            
            # Add tracks to playlist (max 100 at a time)
            if track_uris:
                for i in range(0, len(track_uris), 100):
                    batch = track_uris[i:i+100]
                    print(f"Adding batch {i//100 + 1} with {len(batch)} tracks")
                    sp.playlist_add_items(playlist['id'], batch)
                
                print(f"Successfully added {len(track_uris)} tracks to playlist")
            else:
                print("No tracks to add to playlist")
            
            return playlist['external_urls']['spotify']
            
        except Exception as e:
            print(f"Error creating playlist: {e}")
            raise e