import subprocess
from pathlib import Path
from typing import Optional
from langchain.callbacks.manager import CallbackManagerForChainRun

from langchain.chains.video_captioning.models import AudioModel
from langchain.chains.video_captioning.services.service import Processor
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.document_loaders.assemblyai import TranscriptFormat


class AudioProcessor(Processor):
    def __init__(
        self,
        api_key,
        output_audio_path="output_audio.mp3",
    ):
        self.output_audio_path = Path(output_audio_path)
        self.api_key = api_key

    def process(self, video_file_path: str, run_manager: Optional[CallbackManagerForChainRun] = None) -> list:
        try:
            self.__extract_audio(video_file_path)
            return self.__transcribe_audio()
        finally:
            # Cleanup: Delete the MP3 file after processing
            try:
                self.output_audio_path.unlink()
            except FileNotFoundError:
                pass  # File not found, nothing to delete

    def __extract_audio(self, video_file_path: str):
        # Ensure the directory exists where the output file will be saved
        self.output_audio_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-i",
            video_file_path,
            "-vn",
            "-acodec",
            "mp3",
            self.output_audio_path.as_posix(),
            "-y",  # The '-y' flag overwrites the output file if it exists
        ]

        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

    def __transcribe_audio(self) -> list:
        if not self.api_key:
            raise ValueError("API key for AssemblyAI is not configured")
        audio_file_path_str = str(self.output_audio_path)
        loader = AssemblyAIAudioTranscriptLoader(
            file_path=audio_file_path_str,
            api_key=self.api_key,
            transcript_format=TranscriptFormat.SUBTITLES_SRT,
        )
        docs = loader.load()
        return self.__create_transcript_models(docs)

    @staticmethod
    def __create_transcript_models(docs):
        # Assuming docs is a list of Documents with .page_content as the transcript data
        models = []
        for doc in docs:
            models.extend(AudioProcessor.__parse_transcript(doc.page_content))
        return models

    @staticmethod
    def __parse_transcript(srt_content: str):
        models = []
        entries = srt_content.strip().split("\n\n")  # Split based on double newline

        for entry in entries:
            index, timespan, *subtitle_lines = entry.split("\n")

            # If not a valid entry format, skip
            if len(subtitle_lines) == 0:
                continue

            start_time, end_time = timespan.split(" --> ")
            subtitle_text = " ".join(subtitle_lines).strip()
            models.append(AudioModel.from_srt(start_time, end_time, subtitle_text))
            
        return models
