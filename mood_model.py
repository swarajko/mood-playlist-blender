import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class MoodEmbeddingNet(nn.Module):
    """Deep learning model for mood embedding"""
    def __init__(self, input_dim=9, embedding_dim=16):
        super(MoodEmbeddingNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_dim),
            nn.Tanh()  # Keep embeddings in [-1, 1]
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Output in [0, 1] for features
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction
    
    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

class MoodModel:
    def __init__(self):
        self.model = MoodEmbeddingNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.trained = False
        
    def train_model(self, features, epochs=50):
        """Train autoencoder on user's music features"""
        # Convert to tensor
        X = torch.FloatTensor(features)
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            embeddings, reconstructions = self.model(X)
            
            # Calculate loss (reconstruction + regularization)
            recon_loss = self.criterion(reconstructions, X)
            
            # Add contrastive loss to spread embeddings
            embedding_loss = self._contrastive_loss(embeddings)
            
            total_loss = recon_loss + 0.1 * embedding_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            losses.append(total_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
        
        self.trained = True
        return losses
    
    def _contrastive_loss(self, embeddings):
        """Encourage diversity in embeddings"""
        # Calculate pairwise distances
        distances = torch.cdist(embeddings, embeddings)
        
        # Penalize very similar embeddings (excluding diagonal)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool)
        masked_distances = distances[mask]
        
        # Loss is higher when songs are too similar
        loss = torch.mean(torch.exp(-masked_distances))
        return loss
    
    def get_song_embeddings(self, features):
        """Get embeddings for all songs"""
        self.model.eval()
        X = torch.FloatTensor(features)
        with torch.no_grad():
            embeddings = self.model.get_embedding(X)
        return embeddings.numpy()
    
    def find_similar_songs(self, target_embedding, all_embeddings, exclude_indices=[], top_k=1):
        """Find most similar songs to target embedding"""
        # Calculate similarities
        similarities = cosine_similarity(target_embedding.reshape(1, -1), all_embeddings)[0]
        
        # Exclude already selected songs
        for idx in exclude_indices:
            similarities[idx] = -np.inf
        
        # Get top k similar songs
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return top_indices[0] if top_k == 1 else top_indices
    
    def generate_playlist(self, mood_a_vec, mood_b_vec, features, tracks_df, 
                         n_songs=10, duration_split='equal'):
        """Generate playlist transitioning from mood A to mood B"""
        
        # Get embeddings for all songs
        song_embeddings = self.get_song_embeddings(features)
        
        # Convert mood vectors to embeddings
        mood_a_tensor = torch.FloatTensor(mood_a_vec).unsqueeze(0)
        mood_b_tensor = torch.FloatTensor(mood_b_vec).unsqueeze(0)
        
        with torch.no_grad():
            mood_a_embedding = self.model.get_embedding(mood_a_tensor).numpy()
            mood_b_embedding = self.model.get_embedding(mood_b_tensor).numpy()
        
        # Create interpolation path in embedding space
        playlist_indices = []
        
        # Determine split
        if duration_split == 'short':
            split_a, split_b = 3, 7
        elif duration_split == 'long':
            split_a, split_b = 7, 3
        else:
            split_a, split_b = 5, 5
        
        # Generate playlist
        for i in range(n_songs):
            # Calculate interpolation weight
            if i < split_a:
                weight = 1 - (i / split_a) * 0.5
            else:
                idx_in_b = i - split_a
                weight = 0.5 - (idx_in_b / split_b) * 0.5
            
            # Interpolate in embedding space
            target_embedding = mood_a_embedding * weight + mood_b_embedding * (1 - weight)
            
            # Find closest song
            song_idx = self.find_similar_songs(
                target_embedding, 
                song_embeddings, 
                exclude_indices=playlist_indices
            )
            
            playlist_indices.append(song_idx)
        
        # Get track information
        playlist_tracks = tracks_df.iloc[playlist_indices]
        
        return playlist_tracks
    
    def save_model(self, path='models/mood_model.pkl'):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_state': self.model.state_dict(),
                'trained': self.trained
            }, f)
    
    def load_model(self, path='models/mood_model.pkl'):
        """Load trained model"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.model.load_state_dict(checkpoint['model_state'])
                self.trained = checkpoint['trained']
                return True
        return False