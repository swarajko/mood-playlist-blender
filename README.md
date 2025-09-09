<<<<<<< HEAD
﻿# 🎵 AI Mood Transition Playlist Generator

An intelligent playlist generator that creates smooth mood transitions using AI and your Spotify music library.

## ✨ Features

- 🎯 **Mood-Based Playlists**: Generate playlists that transition between different moods
- 🤖 **AI-Powered**: Uses deep learning to understand your music taste
- 🎨 **Smooth Transitions**: Creates natural progressions between moods
- 💾 **Spotify Integration**: Save playlists directly to your Spotify account
- 🔄 **Smart Fallbacks**: Works even when Spotify API is unavailable

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🛠️ Local Setup

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

## 🔧 Spotify App Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add redirect URI: `http://localhost:8501/callback` (for local) or `https://your-app-name.streamlit.app/callback` (for deployment)
4. Copy Client ID and Client Secret to your secrets file

## 📱 How to Use

1. **Login** with your Spotify account
2. **Load your music library** (saved tracks, playlists, top tracks)
3. **Select moods** for start and end of transition
4. **Choose transition speed** (quick, balanced, or gradual)
5. **Generate playlist** and save to Spotify

## 🎭 Available Moods

- Happy, Sad, Energetic, Calm, Angry
- Romantic, Melancholic, Party, Focus, Chill

## 🏗️ Architecture

- **Frontend**: Streamlit
- **ML Model**: PyTorch-based autoencoder for mood embeddings
- **API**: Spotify Web API
- **Data Processing**: Pandas, NumPy, Scikit-learn

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues, please open an issue on GitHub.

=======
# mood-playlist-blender
AI-powered mood transition playlist generator using Spotify API
>>>>>>> 5a123ffb713726ecab69d740e0faff6436d9e8c3
