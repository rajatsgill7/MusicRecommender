import os
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_playlist_dir: str
    candidate_tracks_dir: str
    top_n: int = 5

class AudioRecommender:
    def __init__(self, user_playlist_dir: str, candidate_tracks_dir: str):
        self.user_playlist_dir = os.path.abspath(user_playlist_dir)
        self.candidate_tracks_dir = os.path.abspath(candidate_tracks_dir)
        self.sr = 22050
        self.user_vectors = []
        self.candidate_vectors = []
        self.candidate_names = []

    def extract_features(self, file_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True, duration=60.0)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            feature_vector = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                np.mean(spectral_contrast, axis=1),
                np.std(spectral_contrast, axis=1),
                [tempo]
            ])
            return feature_vector
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def build_user_profile(self):
        vectors = []
        for file in os.listdir(self.user_playlist_dir):
            if file.lower().endswith(('.mp3', '.wav', '.flac')):
                path = os.path.join(self.user_playlist_dir, file)
                features = self.extract_features(path)
                if features is not None:
                    vectors.append(features)

        if not vectors:
            raise ValueError("No valid audio files in user playlist.")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(vectors)
        self.user_profile = np.mean(scaled, axis=0)

    def process_candidates(self):
        vectors = []
        names = []
        for file in os.listdir(self.candidate_tracks_dir):
            if file.lower().endswith(('.mp3', '.wav', '.flac')):
                path = os.path.join(self.candidate_tracks_dir, file)
                features = self.extract_features(path)
                if features is not None:
                    vectors.append(features)
                    names.append(file)

        if not vectors:
            raise ValueError("No valid candidate audio files.")

        scaler = StandardScaler()
        self.candidate_vectors = scaler.fit_transform(vectors)
        self.candidate_names = names

    def recommend(self, top_n: int = 5) -> List[Tuple[str, float]]:
        if not hasattr(self, 'user_profile'):
            raise ValueError("User profile not built.")
        if not self.candidate_vectors:
            raise ValueError("Candidate vectors not available.")

        sims = cosine_similarity([self.user_profile], self.candidate_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [(self.candidate_names[i], float(sims[i])) for i in top_indices]

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    try:
        recommender = AudioRecommender(
            user_playlist_dir=request.user_playlist_dir,
            candidate_tracks_dir=request.candidate_tracks_dir
        )
        recommender.build_user_profile()
        recommender.process_candidates()
        recommendations = recommender.recommend(top_n=request.top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("TestingPack.Test:app", host="0.0.0.0", port=8000, reload=True)
