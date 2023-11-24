import ffmpeg
import subprocess
from pathlib import Path

from langchain.chains.video_captioning.models import AudioModel
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.document_loaders.assemblyai import TranscriptFormat


class AudioProcessor:
    def __init__(
        self,
        output_audio_path="output_audio.mp3",
        api_key="f50c08e20ecd4544b175953636f0b936",
    ):
        self.output_audio_path = output_audio_path
        self.api_key = api_key

    def process(self, video_file_path: str) -> list:
        audio_file_path = self.extract_audio(video_file_path)
        return self.transcribe_audio(audio_file_path)

    def extract_audio(self, video_file_path: str) -> Path:
        output_audio_path = Path(self.output_audio_path)

        # Ensure the directory exists where the output file will be saved
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-i",
            video_file_path,
            "-vn",
            "-acodec",
            "mp3",
            output_audio_path.as_posix(),
            "-y",  # The '-y' flag overwrites the output file if it exists
        ]

        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return output_audio_path

    def transcribe_audio(self, audio_file_path: Path) -> list:
        if not self.api_key:
            raise ValueError("API key for AssemblyAI is not configured")
        audio_file_path_str = str(audio_file_path)
        loader = AssemblyAIAudioTranscriptLoader(
            file_path=audio_file_path_str,
            api_key=self.api_key,
            transcript_format=TranscriptFormat.SUBTITLES_SRT,
        )
        docs = loader.load()
        return self.create_transcript_models(docs)

    def create_transcript_models(self, docs):
        # Assuming docs is a list of Documents with .page_content as the transcript data
        models = []
        for doc in docs:
            models.extend(self.parse_transcript(doc.page_content))
        return models

    def parse_transcript(self, srt_content: str):
        models = []
        entries = srt_content.strip().split("\n\n")  # Split based on double newline

        for entry in entries:
            index, timespan, *subtitle_lines = entry.split("\n")

            # If not a valid entry format, skip
            if len(subtitle_lines) == 0:
                continue

            start_time, end_time = timespan.split(" --> ")
            subtitle_text = " ".join(subtitle_lines).strip()
            models.append(AudioModel(start_time, end_time, subtitle_text))
        return models
