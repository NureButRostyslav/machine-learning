import os
import wave
from typing import Tuple

from pydub import AudioSegment
from google.cloud import speech
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
INPUT_DIR = "./audio_wav/"
OUTPUT_DIR = "./Transcripts/"


def mp3_to_wav(audio_file_path: str) -> str:
    """Converts an MP3 audio file to WAV format."""
    if audio_file_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(audio_file_path)
        wav_file_path = audio_file_path.replace('.mp3', '.wav')
        sound.export(wav_file_path, format="wav")
        return wav_file_path

    return audio_file_path


def stereo_to_mono(audio_file_path: str) -> None:
    """Converts a stereo audio file to mono format."""
    sound = AudioSegment.from_wav(audio_file_path)
    sound = sound.set_channels(1)
    sound.export(audio_file_path, format="wav")


def frame_rate_channel(audio_file_path: str) -> Tuple[int, int]:
    """Retrieves the frame rate and the number of channels of a WAV file."""
    with wave.open(audio_file_path, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        return frame_rate, channels


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str) -> None:
    """Uploads a file to a Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def delete_blob(bucket_name: str, blob_name: str) -> None:
    """Deletes a specific blob from a Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()


def google_transcribe(audio_file_name: str) -> str:
    """Orchestrates the speech-to-text transcription process."""
    file_path = os.path.join(INPUT_DIR, audio_file_name)

    file_path = mp3_to_wav(file_path)
    wav_file_name = os.path.basename(file_path)

    frame_rate, channels = frame_rate_channel(file_path)
    if channels > 1:
        stereo_to_mono(file_path)

    upload_blob(BUCKET_NAME, file_path, wav_file_name)
    gcs_uri = f"gs://{BUCKET_NAME}/{wav_file_name}"

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US',
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model="latest_long"
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=10000)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + "\n"

    delete_blob(BUCKET_NAME, wav_file_name)

    return transcript


def write_transcripts(transcript_filename: str, transcript: str) -> None:
    """Saves the transcribed text to a local file."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, transcript_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created directory {INPUT_DIR}. Please place your .wav or .mp3 files there.")
    else:
        for audio_file in os.listdir(INPUT_DIR):
            if audio_file.endswith(('.wav', '.mp3')):
                print(f"Processing file: {audio_file}...")

                text_result = google_transcribe(audio_file)

                txt_filename = audio_file.rsplit('.', 1)[0] + '.txt'
                write_transcripts(txt_filename, text_result)

                print(f"Transcription complete! Saved to {OUTPUT_DIR}{txt_filename}")