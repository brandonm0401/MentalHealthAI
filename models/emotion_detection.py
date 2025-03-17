import io
import wave
import torchaudio.functional as F
import torchaudio.transforms as T
import noisereduce as nr
import numpy as np
import torch
import webrtcvad
import matplotlib.pyplot as plt
import sounddevice as sd
import requests


class AudioPreprocessor:
    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate  
        self.vad = webrtcvad.Vad(3)  # High sensitivity for voice detection
        self.deepgram_api_key = "0d7bc60bba92f16be747fd8685dbd93de933b123"
        self.deepgram_url = "https://api.deepgram.com/v1/listen"

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
        """Complete pipeline: Record → Preprocess → Extract Features → Convert to Text"""
        try:
            waveform = self.record_audio()
            waveform = self.resample_audio(waveform, self.target_sample_rate)
            waveform = self.reduce_noise(waveform, self.target_sample_rate)
            features = self.extract_features(waveform)

            print("Feature Extraction Complete!")
            self.visualize_features(waveform, features)

            text = self.speech_to_text(waveform)
            return text, features
        
        except Exception as e:
            print(f"Error during processing: {e}")
            return None, None

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
        """Extract MFCCs, Chroma, Pitch, Energy, and Spectral Centroid features."""
        mfcc_transform = T.MFCC(
            sample_rate=self.target_sample_rate, n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        mfccs = mfcc_transform(waveform)
        mfccs = (mfccs - torch.mean(mfccs)) / (torch.std(mfccs) + 1e-8)  # Normalize

        spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate, n_fft=400, hop_length=160, n_mels=23
        )
        spectrogram = spectrogram_transform(waveform)

        chroma = torch.mean(spectrogram, dim=1)[:12]  # Chroma features from spectrogram

        pitch = F.detect_pitch_frequency(waveform, sample_rate=self.target_sample_rate)
        pitch = (pitch - torch.mean(pitch)) / (torch.std(pitch) + 1e-8)  # Normalize

        energy = torch.sqrt(torch.mean(waveform ** 2))  # Root Mean Square (RMS) Energy

        spectral_centroid = F.spectral_centroid(
            waveform, 
            sample_rate=self.target_sample_rate, 
            n_fft=400, 
            hop_length=160, 
            pad=0,  
            window=torch.hann_window(400),  
            win_length=400
        )
        spectral_centroid = (spectral_centroid - torch.mean(spectral_centroid)) / (torch.std(spectral_centroid) + 1e-8)  # Normalize

        return {
            "MFCCs": mfccs.numpy().squeeze(),
            "Chroma": chroma.numpy().squeeze(),
            "Pitch": pitch.numpy().squeeze(),
            "Energy": energy.item(),
            "Spectral Centroid": spectral_centroid.numpy().squeeze()
        }

    def visualize_features(self, waveform, features):
        """Visualize Waveform, MFCCs, Chroma, Pitch, Energy, and Spectral Centroid."""
        if features is None:
            print("No features to visualize.")
            return

        fig, axs = plt.subplots(6, 1, figsize=(12, 12))

        # Waveform
        axs[0].plot(waveform.numpy().squeeze(), color="blue")
        axs[0].set_title("Waveform")

        # MFCCs (2D)
        axs[1].imshow(features["MFCCs"], aspect="auto", origin="lower", cmap="inferno")
        axs[1].set_title("MFCCs")

        # Chroma (Ensure it's 2D)
        chroma_data = features["Chroma"]
        if chroma_data.ndim == 1:
            chroma_data = chroma_data.reshape(1, -1)  # Convert to 2D
        axs[2].imshow(chroma_data, aspect="auto", origin="lower", cmap="coolwarm")
        axs[2].set_title("Chroma Features")

        # Pitch (1D - Use plot)
        axs[3].plot(features["Pitch"], color="green")
        axs[3].set_title("Pitch (Fundamental Frequency - F0)")

        # Energy (Scalar - Use bar)
        axs[4].bar(["Energy"], [features["Energy"]], color="red")
        axs[4].set_title("Energy (RMS)")

        # Spectral Centroid (1D - Use plot)
        axs[5].plot(features["Spectral Centroid"], color="purple")
        axs[5].set_title("Spectral Centroid")

        plt.tight_layout()
        plt.show()

    def speech_to_text(self, waveform):
        """Convert speech to text using Deepgram API."""
        # Convert waveform to WAV format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(waveform.numpy().astype(np.int16).tobytes())

        audio_data = wav_buffer.getvalue()
        headers = {
            "Authorization": f"Token {self.deepgram_api_key}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(self.deepgram_url, headers=headers, data=audio_data)

        if response.status_code == 200:
            return response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

        print(f"Deepgram Error: {response.status_code} - {response.text}")
        return "Transcription failed."


if __name__ == "__main__":
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    text, features = preprocessor.preprocess_audio()
    if text:
        print(f"Final Transcription: {text}")
