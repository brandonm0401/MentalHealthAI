import io
import wave
import torchaudio.functional as F
import torchaudio.transforms as T
import noisereduce as nr
import numpy as np
import torch
import webrtcvad
import sounddevice as sd
import requests
import joblib
from scipy.stats import skew

class AudioPreprocessor:
    def __init__(self, target_sample_rate=16000, xgb_model_path="xgboost_emotion_classifier.pkl"):
        self.target_sample_rate = target_sample_rate  
        self.vad = webrtcvad.Vad(3)  # High sensitivity for voice detection
        self.deepgram_api_key = "0d7bc60bba92f16be747fd8685dbd93de933b123"
        self.deepgram_url = "https://api.deepgram.com/v1/listen"
        # Load the trained XGBoost model
        self.xgb_model = joblib.load(xgb_model_path)

        # Feature extraction parameters
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = self.n_fft

    def record_audio(self):
        """Record audio and stop automatically when voice activity is not detected."""
        print("Listening for speech...")
        audio_buffer = []
        frame_duration = 0.03  # 30ms per frame
        frame_samples = int(frame_duration * self.target_sample_rate)
        silence_frames = 0
        recording_started = False
        energy_threshold = 5

        with sd.InputStream(samplerate=self.target_sample_rate, channels=1, dtype="int16") as stream:
            while True:
                frame, _ = stream.read(frame_samples)
                frame_np = np.frombuffer(frame, dtype=np.int16)

                is_speech = self.vad.is_speech(frame_np.astype(np.int16).tobytes(), self.target_sample_rate)
                energy = np.sqrt(np.mean(frame_np.astype(np.float32) ** 2))

                if is_speech or energy > energy_threshold:
                    audio_buffer.extend(frame_np)
                    recording_started = True
                    silence_frames = 0  # Reset silence counter
                elif recording_started:
                    silence_frames += 1  # Increment silence counter
                    if silence_frames >= 5:  # Stop recording if no speech is detected for ~150ms
                        print("Silence detected. Stopping recording.")
                        break

        if len(audio_buffer) == 0:
            raise ValueError("No speech detected. Please try again.")

        print("Recording complete!")
        return torch.tensor(audio_buffer, dtype=torch.float32).unsqueeze(0)

    def preprocess_audio(self):
        """Complete pipeline: Record → Preprocess → Extract Features → Convert to Text → Classify Emotion"""
        try:
            waveform = self.record_audio()
            waveform = self.resample_audio(waveform, self.target_sample_rate)
            waveform = self.reduce_noise(waveform, self.target_sample_rate)
            features = self.extract_features(waveform)

            print("Feature Extraction Complete!")

            text = self.speech_to_text(waveform)
            emotion = self.classify_emotion(features)

            return text, emotion, features
        
        except Exception as e:
            print(f"Error during processing: {e}")
            return None, None, None

    def resample_audio(self, waveform, sample_rate):
        """Ensure the waveform is at the target sample rate."""
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform.to(torch.float32))
        return waveform

    def reduce_noise(self, waveform, sample_rate):
        """Apply noise reduction."""
        waveform_np = waveform.numpy().squeeze()
        denoised_waveform_np = nr.reduce_noise(y=waveform_np, sr=sample_rate, prop_decrease=0.9)
        return torch.tensor(denoised_waveform_np, dtype=torch.float32).unsqueeze(0)

    def extract_features(self, waveform):
        """Extract required features for emotion classification."""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        # Define transformations
        mfcc_transform = T.MFCC(sample_rate=self.target_sample_rate, n_mfcc=13, 
                                melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length, "n_mels": 40})
        spectral_centroid_transform = F.spectral_centroid
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=self.target_sample_rate, 
                                                    n_fft=self.n_fft, hop_length=self.hop_length, n_mels=40)

        # Compute MFCCs
        mfcc = mfcc_transform(waveform).squeeze().numpy()
        mfcc_delta = np.diff(mfcc, axis=1)  # First-order difference
        mfcc_delta2 = np.diff(mfcc_delta, axis=1)  # Second-order difference

        # Compute Spectral Centroid
        window = torch.hann_window(self.win_length)
        spectral_centroid = spectral_centroid_transform(waveform, sample_rate=self.target_sample_rate,
                                                        pad=0, window=window, n_fft=self.n_fft, hop_length=self.hop_length, 
                                                        win_length=self.win_length)
        spectral_centroid = spectral_centroid.squeeze().numpy()

        # Compute Mel Spectrogram & Chroma
        mel_spec = mel_spectrogram_transform(waveform)
        chroma = torch.log(mel_spec + 1e-6).mean(dim=1).numpy()

        # Compute RMS Energy
        energy = torch.sqrt(torch.mean(waveform**2)).item()

        # Compute Pitch
        pitch = F.detect_pitch_frequency(waveform, sample_rate=self.target_sample_rate).squeeze().numpy()
        pitch_delta = np.diff(pitch) if len(pitch) > 1 else np.array([0])

        # Return Features as a Dictionary ✅
        features = {
            "mfcc_mean": np.mean(mfcc),
            "mfcc_delta_mean": np.mean(mfcc_delta),
            "mfcc_delta2_mean": np.mean(mfcc_delta2),
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_centroid_std": np.std(spectral_centroid),
            "spectral_centroid_var": np.var(spectral_centroid),
            "chroma_mean": np.mean(chroma),
            "chroma_delta_mean": np.mean(np.diff(chroma)),
            "chroma_skew": skew(chroma.flatten()),
            "energy": energy,
            "waveform_peak": np.max(waveform.numpy()),
            "waveform_var": np.var(waveform.numpy()),
            "pitch_mean": np.mean(pitch),
            "pitch_delta_mean": np.mean(pitch_delta),
            "pitch_delta_std": np.std(pitch_delta),
        }
        print("Extracted Features:", features)  # Debugging
        return features

    def classify_emotion(self, features):
        FEATURE_ORDER = [
            "mfcc_mean", "mfcc_delta_mean", "mfcc_delta2_mean",
            "spectral_centroid_mean", "spectral_centroid_std", "spectral_centroid_var",
            "chroma_mean", "chroma_delta_mean", "chroma_skew",
            "energy", "waveform_peak", "waveform_var",
            "pitch_mean", "pitch_delta_mean", "pitch_delta_std"
        ]
        label_map = {0: "Happy", 1: "Sad", 2: "Neutral"}
        
        # Convert features into an array (excluding Energy_Variance)
        feature_vector = np.array([features[key] for key in FEATURE_ORDER]).reshape(1, -1)
        
        # Predict emotion using XGBoost model
        predicted_label = self.xgb_model.predict(feature_vector)[0]
        return label_map.get(predicted_label, "Unknown")

if __name__ == "__main__":
    preprocessor = AudioPreprocessor(target_sample_rate=16000, xgb_model_path="xgboost_emotion_classifier.pkl")
    text, emotion, features = preprocessor.preprocess_audio()
    
    if text:
        print(f"Final Transcription: {text}")
    if emotion:
        print(f"Predicted Emotion: {emotion}")
