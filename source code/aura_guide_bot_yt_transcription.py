
"""aura_guide_bot_yt_transcription.ipynb



Transcribing 2 Youtube Videos of Velana using Whisper.
"""

!pip install git+https://github.com/openai/whisper.git
!pip install yt-dlp

import os
import subprocess
import whisper
from typing import List, Dict, Any

def create_audio_directory(directory: str = "yt_audio") -> None:
    """Create a directory for storing audio files if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def download_audio(url: str, output_dir: str = "yt_audio") -> str:
    """Download audio from YouTube video using yt-dlp."""
    try:
        video_id = url.split("v=")[-1].split("?")[0] if "v=" in url else url.split("youtu.be/")[1].split("?")[0]
        output_path = f"{output_dir}/{video_id}.mp3"
        subprocess.run([
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "mp3",
            "-o", output_path,
            url
        ], check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as error:
        print(f"Failed to download audio for {url}: {str(error)}")
        return ""

def transcribe_audio(audio_path: str, model: whisper.Whisper, url: str) -> Dict[str, Any]:
    """Transcribe audio file using Whisper model."""
    try:
        result = model.transcribe(audio_path)
        return {
            "page_content": result["text"],
            "metadata": {"source": url},
            "status": True
        }
    except Exception as error:
        print(f"Whisper transcription failed for {url}: {str(error)}")
        return {"status": False}

def save_transcript(url: str, transcript: str, output_dir: str = "transcripts") -> None:
    """Save transcript to a text file in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    video_id = url.split("v=")[-1].split("?")[0] if "v=" in url else url.split("youtu.be/")[1].split("?")[0]
    output_file = f"{output_dir}/{video_id}.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcript)
        print(f"Saved transcript for {url} to {output_file}")
    except IOError as error:
        print(f"Failed to save transcript for {url}: {str(error)}")

def main():
    """Main function to transcribe YouTube videos using Whisper and save transcripts."""
    # Create directory for audio files
    create_audio_directory()

    # Load Whisper model
    model = whisper.load_model("base")

    # List of YouTube URLs to transcribe
    youtube_urls: List[str] = [
        "https://youtu.be/iMncZjjeBQg?si=dJylmRKj0ff-2Xgm",
        "https://youtu.be/cSFSZgOqpUA?si=fBUa1YBYHTWYSVg0"
    ]

    yt_docs: List[Dict[str, Any]] = []
    session_transcripts: Dict[str, str] = {}

    for url in youtube_urls:
        transcribed = False
        # Download audio
        audio_path = download_audio(url)
        if audio_path and os.path.exists(audio_path):
            # Transcribe with Whisper
            result = transcribe_audio(audio_path, model, url)
            if result.get("status"):
                yt_docs.append({
                    "page_content": result["page_content"],
                    "metadata": result["metadata"]
                })
                session_transcripts[url] = result["page_content"]
                print(f"Successfully transcribed {url} with Whisper: {result['page_content'][:200]}...")
                # Save transcript to file
                save_transcript(url, result["page_content"])
                transcribed = True

        print(f"Transcription status for {url}: {'Transcribed' if transcribed else 'Not transcribed'}")

    # Print saved transcripts summary
    print("\nTranscripts saved in session for URLs:")
    for url, transcript in session_transcripts.items():
        print(f"{url}: {len(transcript)} characters saved")

if __name__ == "__main__":
    main()