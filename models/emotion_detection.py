import io, wave
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
    def __init__(self, target_sample_rate=16000, silence_threshold=2):
        self.target_sample_rate = target_sample_rate  
        self.silence_threshold = silence_threshold  
        self.vad = webrtcvad.Vad(3)
        self.deepgram_api_key = "e7b267d395b28a3f7d6495f6883abb601151042e"
        self.deepgram_url = "https://api.deepgram.com/v1/listen"

    def record_audio(self):
        """Record audio until sustained silence is detected."""
        print("Listening for speech...")
        audio_buffer = []
        frame_duration = 0.03
        frame_samples = int(frame_duration * self.target_sample_rate)
        silence_frames_required = int(self.silence_threshold / frame_duration)
        silence_buffer = 0  
        recording_started = False
        energy_threshold = 500

        with sd.InputStream(samplerate=self.target_sample_rate, channels=1, dtype="int16") as stream:
            while True:
                frame, _ = stream.read(frame_samples)
                frame_np = np.frombuffer(frame, dtype=np.int16)

                is_speech = self.vad.is_speech(frame.tobytes(), self.target_sample_rate)
                energy = np.sqrt(np.mean(frame_np.astype(np.float32) ** 2))

                if is_speech or energy > energy_threshold:
                    audio_buffer.extend(frame_np)
                    recording_started = True
                    silence_buffer = 0  # Reset silence counter
                elif recording_started:
                    silence_buffer += 1  # Increase silence count
                    if silence_buffer >= silence_frames_required:
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
        mfccs = (mfcc_transform(waveform) - torch.mean(mfcc_transform(waveform))) / torch.std(mfcc_transform(waveform))

        spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate, n_fft=400, hop_length=160, n_mels=23
        )
        spectrogram = spectrogram_transform(waveform)

        chroma = spectrogram[:, :12, :]

        pitch = F.detect_pitch_frequency(waveform, sample_rate=self.target_sample_rate)
        pitch = (pitch - torch.mean(pitch)) / (torch.std(pitch) + 1e-8)  # Normalize

        energy_frames = [
            np.sqrt(np.mean(waveform.numpy().squeeze()[i: i + 400] ** 2))
            for i in range(0, len(waveform.numpy().squeeze()) - 400, 160)
        ]
        energy = np.mean(energy_frames)

        # **Spectral Centroid Calculation**
        spectral_centroid = F.spectral_centroid(waveform, sample_rate=self.target_sample_rate, n_fft=400, hop_length=160)
        spectral_centroid = (spectral_centroid - torch.mean(spectral_centroid)) / (torch.std(spectral_centroid) + 1e-8)  # Normalize

        return {
            "MFCCs": mfccs,
            "Chroma": chroma,
            "Pitch": pitch,
            "Energy": energy,
            "Spectral Centroid": spectral_centroid
        }

    def visualize_features(self, waveform, features):
        """Visualize Waveform, MFCCs, Chroma, Pitch, Energy, and Spectral Centroid."""
        if features is None:
            print("No features to visualize.")
            return

        mfccs = features["MFCCs"].squeeze().numpy()
        chroma = features["Chroma"].squeeze().numpy()
        pitch = features["Pitch"].squeeze().numpy()
        energy = features["Energy"]
        spectral_centroid = features["Spectral Centroid"].squeeze().numpy()

        fig, axs = plt.subplots(6, 1, figsize=(12, 12))

        axs[0].plot(waveform.numpy().squeeze(), color="blue")
        axs[0].set_title("Waveform")
        axs[0].set_xlabel("Samples")
        axs[0].set_ylabel("Amplitude")

        im1 = axs[1].imshow(mfccs, aspect="auto", origin="lower", cmap="inferno")
        axs[1].set_title("MFCCs")
        axs[1].set_xlabel("Time Frames")
        axs[1].set_ylabel("MFCC Coefficients")
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(chroma, aspect="auto", origin="lower", cmap="coolwarm")
        axs[2].set_title("Chroma Features")
        axs[2].set_xlabel("Time Frames")
        axs[2].set_ylabel("Chroma Bins")
        fig.colorbar(im2, ax=axs[2])

        axs[3].plot(pitch, color="green")
        axs[3].set_title("Pitch (Fundamental Frequency - F0)")
        axs[3].set_xlabel("Time Frames")
        axs[3].set_ylabel("Frequency (Hz)")

        axs[4].bar(["Energy"], [energy], color="red")
        axs[4].set_title("Energy (RMS)")
        axs[4].set_ylabel("Energy Level")

        axs[5].plot(spectral_centroid, color="purple")
        axs[5].set_title("Spectral Centroid")
        axs[5].set_xlabel("Time Frames")
        axs[5].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    preprocessor = AudioPreprocessor(target_sample_rate=16000, silence_threshold=2)
    text, features = preprocessor.preprocess_audio()

    if text:
        print(f"Final Transcription: {text}")
