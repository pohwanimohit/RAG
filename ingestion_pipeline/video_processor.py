import subprocess
from pathlib import Path

from faster_whisper import WhisperModel
from langchain_core.documents import Document


def process_video(video_path: str):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )
    audio_path = video_path.parent / "audio.wav"
    if not audio_path.exists(): raise FileNotFoundError( f"Audio file not found: {audio_path}" )
    print(f"\n[VIDEO] Processing: {video_path.name}")

    
    print("[VIDEO] Extracting audio...")
    print(f"[VIDEO] Using existing audio: {audio_path}")

    # command = [
    #     "ffmpeg",
    #     "-y",
    #     "-i", str(video_path),
    #     "-vn",
    #     "-ac", "1",
    #     "-ar", "16000",
    #     "-c:a", "pcm_s16le",
    #     str(audio_path),
    # ]

    # subprocess.run(
    #     command,
    #     check=True,
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.PIPE,
    # )

    # print(f"[VIDEO] Audio saved: {audio_path}")

    # Loading the Whisper model.
    print("[VIDEO] Loading Whisper model...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    #Transcribing the audio.
    print("[VIDEO] Transcribing audio...")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
    )

    transcript_parts = []

    for segment in segments:
        transcript_parts.append(segment.text.strip())

    transcript = " ".join(transcript_parts)

    print(f"[VIDEO] Detected language: {info.language}")
    print(f"[VIDEO] Transcript:\n{transcript}")
    print(
        f"[VIDEO] Transcript characters: {len(transcript)}"
    )

    # Converting the transcript into a LangChain Document.
    video_document = Document(
        page_content=transcript,
        metadata={
            "source": str(video_path),
            "language": info.language,
            "type": "video_transcript",
        },
    )

    return [video_document]