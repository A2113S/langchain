from langchain.chains.video_captioning.models import (
    AudioModel,
    VideoModel,
    CaptionModel,
)
from langchain.chains.video_captioning.prompt import VIDEO_AUDIO_VALIDATING_PROMPT


class CombineProcessor:
    def __init__(self, llm, verbose):
        self.llm = llm
        self.verbose = verbose

    def process(self, audio_models, video_models) -> list:
        from langchain.chains import LLMChain

        def validate_and_adjust_description(
            audio_model: AudioModel, video_model: VideoModel
        ):
            validation = LLMChain(
                llm=self.llm, prompt=VIDEO_AUDIO_VALIDATING_PROMPT, verbose=self.verbose
            )
            response = validation(
                {
                    "subtitle": audio_model.get_subtitle_text(),
                    "image_description": video_model.get_image_description(),
                }
            )
            return response["text"].replace("Result:", "").strip()

        def milliseconds_to_srt_time(ms):
            if isinstance(ms, str) and "," in ms:
                return ms

            """Converts milliseconds to SRT time format 'HH:MM:SS,mmm'."""
            hours = int(ms // 3600000)
            minutes = int((ms % 3600000) // 60000)
            seconds = int((ms % 60000) // 1000)
            milliseconds = int(ms % 1000)

            return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

        def find_overlapping_audio_models(video_model, audio_models):
            overlapping_models = []
            video_start = video_model.get_start_time()
            video_end = video_model.get_end_time()

            for audio_model in audio_models:
                audio_start = self.parse_time(str(audio_model.get_start_time()))
                audio_end = self.parse_time(str(audio_model.get_end_time()))
                overlap_start = max(audio_start, video_start)
                overlap_end = min(audio_end, video_end)

                if overlap_start < overlap_end:
                    overlapping_models.append(audio_model)

            return overlapping_models

        caption_models = []

        for video_model in video_models:
            overlapping_audio_models = find_overlapping_audio_models(
                video_model, audio_models
            )

            if overlapping_audio_models:
                # Create separate captions for each overlapping audio model
                last_end_time = video_model.get_start_time()  # Initialize last_end_time

                for audio_model in overlapping_audio_models:
                    overlap_start = max(
                        video_model.get_start_time(),
                        self.parse_time(audio_model.get_start_time()),
                    )
                    overlap_end = min(
                        video_model.get_end_time(),
                        self.parse_time(audio_model.get_end_time()),
                    )

                    # Create a caption for the overlapping period
                    caption_text = f"[{validate_and_adjust_description(audio_model, video_model)}] {audio_model.get_subtitle_text()}"
                    caption_model = CaptionModel(
                        milliseconds_to_srt_time(overlap_start),
                        milliseconds_to_srt_time(overlap_end),
                        caption_text,
                    )
                    caption_models.append(caption_model)

                    last_end_time = overlap_end  # Update last_end_time

                # If there is a gap after the last overlapping period, create a caption for that gap
                if last_end_time < video_model.get_end_time():
                    gap_start = last_end_time
                    gap_end = video_model.get_end_time()
                    gap_caption_text = f"[{video_model.get_image_description()}]"
                    gap_caption_model = CaptionModel(
                        milliseconds_to_srt_time(gap_start),
                        milliseconds_to_srt_time(gap_end),
                        gap_caption_text,
                    )
                    caption_models.append(gap_caption_model)

            else:
                # No overlapping audio, use video model's description for the entire duration
                caption_text = f"[{video_model.get_image_description()}]"
                caption_model = CaptionModel(
                    self.milliseconds_to_srt_time(video_model.get_start_time()),
                    self.milliseconds_to_srt_time(video_model.get_end_time()),
                    caption_text,
                )
                caption_models.append(caption_model)

        return caption_models

    def parse_time(self, s):
        """Converts a time string into milliseconds."""
        h, m, s = map(float, s.replace(",", ".").split(":"))
        return int((h * 3600 + m * 60 + s) * 1000)
