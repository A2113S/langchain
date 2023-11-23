import ffmpeg

from langchain.chains.video_captioning.models import AudioModel
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.document_loaders.assemblyai import TranscriptFormat


class AudioProcessor:
    def process(self, video_file_path: str) -> str:
        audio_file_path = self.extract_audio(video_file_path)
        transcription = self.transcribe_audio(audio_file_path)
        return transcription

    def extract_audio(self, video_file_path: str) -> str:
        audio_file_path = "output_audio.mp3"
        (ffmpeg.input(video_file_path).output(audio_file_path, format="mp3").run())
        return audio_file_path

    def CreateTranscriptModels(self, doc):
        subtitles = doc.strip().split(
            "\n\n"
        )  # Splitting based on double newline, which separates SRT entries
        models = []

        for subtitle in subtitles:
            lines = subtitle.split("\n")
            if (
                len(lines) >= 3
            ):  # Checking if there are enough lines for an index, timestamp, and text
                times = lines[1].split(" --> ")
                start_time = times[0].strip()
                end_time = times[1].strip()

                subtitle_text = " ".join(lines[2:]).strip()

                transcript_model = AudioModel(start_time, end_time, subtitle_text)
                models.append(transcript_model)

        return models

    def transcribe_audio(self, audio_file_path: str) -> str:
        loader = AssemblyAIAudioTranscriptLoader(
            file_path=audio_file_path,
            api_key="your_api_key",  # Replace with your actual API key
            transcript_format=TranscriptFormat.SUBTITLES_SRT,
        )

        # Load the transcript
        docs = loader.load()

        # Process the transcript to create AudioModel instances
        all_audio_models = [
            self.CreateTranscriptModels(doc.page_content) for doc in docs
        ]

        # Flatten the list of AudioModel instances
        audio_models = [model for sublist in all_audio_models for model in sublist]

        # Combine the text from all AudioModel instances for the final transcription
        transcription = " ".join([model.text for model in audio_models])

        return transcription
