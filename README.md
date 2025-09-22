🎵 AI Mood Transition Playlist Generator

An intelligent playlist generator that creates smooth mood transitions using AI and your Spotify music library.

## Features

- 🎯 **Mood-Based Playlists**: Generate playlists that transition between different moods
- 🤖 **AI-Powered**: Uses deep learning to understand your music taste
- 🎨 **Smooth Transitions**: Creates natural progressions between moods
- 💾 **Spotify Integration**: Save playlists directly to your Spotify account
- 🔄 **Smart Fallbacks**: Works even when Spotify API is unavailable

## Local Setup

### Prerequisites
- Python 3.8+
- Spotify Developer Account
- Spotify App credentials

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/mood-playlist-generator.git
cd mood-playlist-generator
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Spotify credentials:**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Fill in your Spotify app credentials

5. **Run the app:**
```bash
streamlit run app.py
```

## Spotify App Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add redirect URI: `http://localhost:8501/callback`
4. Copy Client ID and Client Secret to your secrets file

## How to Use

1. **Login** with your Spotify account
2. **Load your music library** (saved tracks, playlists, top tracks)
3. **Select moods** for start and end of transition
4. **Choose transition speed** (quick, balanced, or gradual)
5. **Generate playlist** and save to Spotify

## Available Moods

- Happy, Sad, Energetic, Calm, Angry
- Romantic, Melancholic, Party, Focus, Chill

## Architecture

- **Frontend**: Streamlit
- **ML Model**: PyTorch-based autoencoder for mood embeddings
- **API**: Spotify Web API
- **Data Processing**: Pandas, NumPy, Scikit-learn

## Deep Learning Model

**PyTorch Autoencoder with Contrastive Loss:**
- **Input**: 9 audio features (danceability, energy, valence, etc.)
- **Encoder**: 9 → 64 → 32 → 16 (mood embedding)
- **Decoder**: 16 → 32 → 64 → 9 (reconstruction)
- **Loss**: Reconstruction + Contrastive loss for diverse embeddings
- **Training**: Unsupervised on user's music library

## License

MIT License - see LICENSE file for details

