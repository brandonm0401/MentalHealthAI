import requests

DEEPGRAM_API_KEY = "e7b267d395b28a3f7d6495f6883abb601151042e"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

headers = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "audio/wav"
}

# Replace with the actual file path to your audio
audio_file_path = "recording.wav"

def transcribe_audio(audio_file_path):
    """Send audio to Deepgram API and extract the transcript as plain text."""
    with open(audio_file_path, "rb") as audio:
        response = requests.post(DEEPGRAM_URL, headers=headers, data=audio)

    try:
        response_json = response.json()
        alternatives = response_json.get("results", {}).get("channels", [{}])[0].get("alternatives", [])

        if not alternatives:
            return "⚠️ No transcription found."

        # Extract the transcript
        transcript = alternatives[0].get("transcript", "").strip()
        confidence = alternatives[0].get("confidence", 0)

        if transcript and confidence >= 0.5:  # Adjust confidence threshold if needed
            return transcript
        else:
            return "⚠️ Low-confidence transcript. No reliable text extracted."

    except Exception as e:
        return f"⚠️ Error processing response: {e}"

# Get transcript
transcript_text = transcribe_audio(audio_file_path)
print("Transcribed Text:", transcript_text)

#e7b267d395b28a3f7d6495f6883abb601151042e