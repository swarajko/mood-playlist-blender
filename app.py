import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from spotify_client import SpotifyClient
from data_processor import DataProcessor
from mood_model import MoodModel
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Mood Transition Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'token_info' not in st.session_state:
    st.session_state.token_info = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Initialize clients
spotify_client = SpotifyClient()
data_processor = DataProcessor()

def authenticate_spotify():
    """Handle Spotify authentication"""
    auth_manager = spotify_client.create_oauth()
    
    # Check if we have a code in the URL
    params = st.experimental_get_query_params()
    
    if 'code' in params:
        # Exchange code for token
        try:
            token_info = auth_manager.get_access_token(params['code'][0])
            if token_info:
                st.session_state.token_info = token_info
                st.session_state.authenticated = True
                # Clear the URL parameters
                st.experimental_set_query_params()
                st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.experimental_set_query_params()
    else:
        # Show login button
        auth_url = auth_manager.get_authorize_url()
        st.markdown(f"[🔗 Login with Spotify]({auth_url})")

def load_user_data():
    """Load or fetch user's music data"""
    if st.session_state.user_data is not None:
        return st.session_state.user_data
    
    sp, token_info = spotify_client.get_spotify_and_token(st.session_state.token_info)
    if sp is None:
        st.error("Authentication expired. Please login again.")
        st.session_state.authenticated = False
        return None
    # persist refreshed token if changed
    if token_info != st.session_state.token_info:
        st.session_state.token_info = token_info
    user_id = sp.current_user()['id']
    
    # Check if we have cached data
    cached_data = data_processor.load_user_data(user_id)
    
    if cached_data is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📁 Found cached data with {len(cached_data)} tracks")
        with col2:
            if st.button("🔄 Refresh My Music Library"):
                # Clear cached data and fetch fresh
                st.session_state.user_data = None
                st.rerun()
        st.session_state.user_data = cached_data
        return cached_data
    
    # Fetch new data
    with st.spinner("Fetching your music library... This may take a minute."):
        tracks = spotify_client.get_user_tracks(sp)
        st.info(f"Found {len(tracks)} unique tracks across Saved, Top, and Playlists.")
        if len(tracks) < 10:
            st.error("Not enough tracks found. Please add more music to your Spotify library.")
            return None
        
        track_ids = [t['id'] for t in tracks]
        audio_features = spotify_client.get_audio_features(sp, track_ids)
        
        # Always ensure we have features (synthetic fallback should handle this)
        if len(audio_features) == 0:
            st.warning("⚠️ Spotify API returned no features. Generating synthetic features...")
            # Force synthetic features generation
            audio_features = spotify_client._generate_synthetic_features(track_ids)
            st.success(f"✅ Generated {len(audio_features)} synthetic audio features.")
        else:
            st.success(f"✅ Fetched {len(audio_features)} audio features from Spotify.")
        
        st.info(f"🎵 Using {len(audio_features)} audio features for playlist generation.")
        
        # Process and save
        user_data = data_processor.process_and_save_tracks(tracks, audio_features, user_id)
        if user_data is None or user_data.empty:
            st.error("Couldn't process tracks. Please try again.")
            return None
        st.session_state.user_data = user_data
        
        st.success(f"🎵 Loaded {len(user_data)} tracks from your library!")
        
    return user_data

def train_model(user_data):
    """Train or load the mood model"""
    if st.session_state.model is not None:
        return st.session_state.model
    
    model = MoodModel()
    
    # Try to load existing model
    sp, token_info = spotify_client.get_spotify_and_token(st.session_state.token_info)
    if sp is None:
        st.error("Authentication expired. Please login again.")
        st.session_state.authenticated = False
        return None
    if token_info != st.session_state.token_info:
        st.session_state.token_info = token_info
    user_id = sp.current_user()['id']
    model_path = f'models/mood_model_{user_id}.pkl'
    
    if os.path.exists(model_path):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("🧠 Found existing trained model")
        with col2:
            if st.button("🔄 Retrain Model"):
                # Force retrain by removing cached model
                st.session_state.model = None
                st.rerun()
        model.load_model(model_path)
        st.session_state.model = model
        return model
    
    # Train new model
    with st.spinner("Training mood model on your music taste..."):
        features = data_processor.normalize_features(user_data)
        
        # Show training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train model
        model.train_model(features, epochs=30)
        
        # Update progress
        for i in range(101):
            progress_bar.progress(i)
            if i % 20 == 0:
                status_text.text(f"Training... {i}% complete")
            time.sleep(0.01)
        
        # Save model
        model.save_model(model_path)
        st.session_state.model = model
        
        status_text.text("Model training complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
    
    return model

def visualize_mood_transition(mood_a, mood_b, duration_split):
    """Create visualization of mood transition"""
    fig = go.Figure()
    
    # Create gradient effect
    if duration_split == 'short':
        colors = ['rgb(255,0,0)'] * 3 + ['rgb(200,50,50)'] * 2 + \
                ['rgb(150,100,100)'] * 2 + ['rgb(0,0,255)'] * 3
    elif duration_split == 'long':
        colors = ['rgb(255,0,0)'] * 7 + ['rgb(150,100,100)'] * 2 + ['rgb(0,0,255)'] * 1
    else:
        colors = ['rgb(255,0,0)'] * 5 + ['rgb(0,0,255)'] * 5
    
    # Add bars
    fig.add_trace(go.Bar(
        x=list(range(1, 11)),
        y=[1] * 10,
        marker_color=colors,
        text=[f'Song {i}' for i in range(1, 11)],
        textposition='inside',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Mood Transition: {mood_a} → {mood_b}",
        xaxis_title="Track Position",
        yaxis_visible=False,
        height=200,
        margin=dict(t=50, b=30, l=0, r=0)
    )
    
    return fig

def main():
    st.title("🎵 AI Mood Transition Playlist Generator")
    st.markdown("Create playlists that smoothly transition between moods using AI")
    
    # Sidebar for authentication
    with st.sidebar:
        st.header("🔐 Authentication")
        
        if not st.session_state.authenticated:
            st.info("Please login with Spotify to continue")
            authenticate_spotify()
            return
        else:
            sp, token_info = spotify_client.get_spotify_and_token(st.session_state.token_info)
            if sp is None:
                st.info("Session expired. Please login again.")
                authenticate_spotify()
                return
            if token_info != st.session_state.token_info:
                st.session_state.token_info = token_info
            user = sp.current_user()
            st.success(f"Logged in as: {user['display_name']}")
            
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.token_info = None
                st.session_state.user_data = None
                st.session_state.model = None
                st.rerun()
    
    # Main content
    if st.session_state.authenticated:
        # Load user data
        user_data = load_user_data()
        
        if user_data is None:
            return
        
        # Train model
        model = train_model(user_data)
        
        st.markdown("---")
        
        # Playlist generation interface
        st.header("🎨 Create Your Mood Transition Playlist")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mood_a = st.selectbox(
                "Starting Mood",
                ['Happy', 'Sad', 'Energetic', 'Calm', 'Angry', 
                 'Romantic', 'Melancholic', 'Party', 'Focus', 'Chill']
            )
        
        with col2:
            mood_b = st.selectbox(
                "Ending Mood",
                ['Happy', 'Sad', 'Energetic', 'Calm', 'Angry', 
                 'Romantic', 'Melancholic', 'Party', 'Focus', 'Chill']
            )
        
        with col3:
            duration_split = st.radio(
                "Transition Speed",
                ['short', 'equal', 'long'],
                format_func=lambda x: {
                    'short': 'Quick (3:7)',
                    'equal': 'Balanced (5:5)',
                    'long': 'Gradual (7:3)'
                }[x]
            )
        
        # Visualize transition
        fig = visualize_mood_transition(mood_a, mood_b, duration_split)
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate playlist button
        if st.button("🎵 Generate Playlist", type="primary"):
            with st.spinner("Creating your perfect mood transition playlist..."):
                # Get mood vectors
                mood_a_vec = data_processor.get_mood_vector(mood_a)
                mood_b_vec = data_processor.get_mood_vector(mood_b)
                
                # Get normalized features
                features = data_processor.normalize_features(user_data)
                
                # Generate playlist
                playlist = model.generate_playlist(
                    mood_a_vec, mood_b_vec, 
                    features, user_data,
                    n_songs=10,
                    duration_split=duration_split
                )
                
                # Show the resulting tracks
                st.subheader("Your Mood Transition Playlist")
                try:
                    st.dataframe(playlist[["name", "artist", "album"]].reset_index(drop=True))
                except Exception:
                    st.dataframe(playlist.reset_index(drop=True))

                # Save playlist to Spotify
                if st.button("💾 Save to Spotify", key="save_playlist"):
                    with st.spinner("Creating playlist in Spotify..."):
                        try:
                            sp, token_info = spotify_client.get_spotify_and_token(st.session_state.token_info)
                            if sp is None:
                                st.error("❌ Authentication expired. Please login again.")
                                return
                            if token_info != st.session_state.token_info:
                                st.session_state.token_info = token_info
                            
                            # Test user access first
                            user = sp.current_user()
                            st.info(f"✅ Connected as: {user['display_name']}")
                            
                            playlist_name = f"{mood_a} → {mood_b} Mood Transition"
                            
                            # Debug: Show playlist data
                            st.write("🔍 Debug - Playlist data:")
                            st.write(f"Number of tracks: {len(playlist)}")
                            st.write(f"Columns: {list(playlist.columns)}")
                            
                            # Ensure we have URIs - convert track IDs to URIs if needed
                            track_uris = []
                            for idx, track in playlist.iterrows():
                                if 'uri' in track and pd.notna(track['uri']) and track['uri']:
                                    track_uris.append(track['uri'])
                                elif 'id' in track and pd.notna(track['id']) and track['id']:
                                    # Convert track ID to URI format
                                    track_uris.append(f"spotify:track:{track['id']}")
                                else:
                                    st.warning(f"⚠️ Skipping track {idx} - no valid URI or ID")
                            
                            st.write(f"📝 Valid track URIs: {len(track_uris)}")
                            
                            if not track_uris:
                                st.error("❌ No valid track URIs found for playlist creation.")
                                return
                            
                            # Create playlist
                            playlist_url = spotify_client.create_playlist(sp, playlist_name, track_uris)
                            st.success(f"🎉 Playlist '{playlist_name}' saved to Spotify!")
                            st.markdown(f"**[🔗 Open Playlist in Spotify]({playlist_url})**")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to save playlist: {str(e)}")
                            st.write("🔍 Full error details:")
                            st.code(str(e))
                            st.info("💡 **Troubleshooting:** Make sure you have permission to create playlists in your Spotify account and that your Spotify app has the correct scopes.")

if __name__ == "__main__":
    main()
