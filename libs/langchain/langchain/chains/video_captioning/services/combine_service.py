from typing import Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.chains.video_captioning.models import (
    AudioModel,
    VideoModel,
    CaptionModel,
)
from langchain.chains.video_captioning.prompts import VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT
from langchain.schema.language_model import BaseLanguageModel


class CombineProcessor:
    def __init__(self, llm: BaseLanguageModel, verbose = True, char_limit: int = 20):
        self.llm = llm
        self.verbose = verbose

        # Adjust as needed. Be careful adjusting it too low because OpenAI may produce unwanted output
        self.__CHAR_LIMIT = char_limit


    # Helper
    def __parse_time(s:str) -> int:
        """Converts a time string into milliseconds."""
        h, m, s = map(float, s.replace(',', '.').split(':'))
        return int((h * 3600 + m * 60 + s) * 1000)

    # Helper
    def __milliseconds_to_srt_time(ms:int,str) -> str:    
        if isinstance(ms, str) and ',' in ms:        
            return ms

        """Converts milliseconds to SRT time format 'HH:MM:SS,mmm'."""
        hours = int(ms // 3600000)
        minutes = int((ms % 3600000) // 60000)
        seconds = int((ms % 60000) // 1000)
        milliseconds = int(ms % 1000)

        return f'{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}'
    
    
    def process(self, video_models: List[VideoModel], audio_models: List[AudioModel]) -> List[CaptionModel]:
        caption_models = []

        for video_model in video_models:
            overlapping_audio_models = self.__find_overlapping_audio_models(video_model, audio_models)

            if overlapping_audio_models:
                # Create separate captions for each overlapping audio model
                last_end_time = video_model.start_time()  # Initialize last_end_time

                for audio_model in overlapping_audio_models:
                    overlap_start = max(video_model.start_time(), CombineProcessor.__parse_time(audio_model.start_time()))
                    overlap_end = min(video_model.end_time(), CombineProcessor.__parse_time(audio_model.end_time()))

                    # Create a caption for the overlapping period
                    caption_text = f"[{self.__validate_and_adjust_description(audio_model, video_model)}] {audio_model.subtitle_text()}"
                    caption_model = CaptionModel(CombineProcessor.__milliseconds_to_srt_time(overlap_start), CombineProcessor.__milliseconds_to_srt_time(overlap_end), caption_text)
                    caption_models.append(caption_model)

                    last_end_time = overlap_end  # Update last_end_time

                # If there is a gap after the last overlapping period, create a caption for that gap
                if last_end_time < video_model.end_time():
                    gap_start = last_end_time
                    gap_end = video_model.end_time()
                    gap_caption_text = f"[{video_model.image_description()}]"
                    gap_caption_model = CaptionModel(CombineProcessor.__milliseconds_to_srt_time(gap_start), CombineProcessor.__milliseconds_to_srt_time(gap_end), gap_caption_text)
                    caption_models.append(gap_caption_model)

            else:
                # No overlapping audio, use video model's description for the entire duration
                caption_text = f"[{video_model.image_description()}]"
                caption_model = CaptionModel(CombineProcessor.__milliseconds_to_srt_time(video_model.start_time()), CombineProcessor.__milliseconds_to_srt_time(video_model.end_time()), caption_text)
                caption_models.append(caption_model)

        return caption_models

    def __find_overlapping_audio_models(self, video_model: VideoModel, audio_models: List[AudioModel]) -> List[AudioModel]:
        overlapping_models = []
        video_start = video_model.start_time()
        video_end = video_model.end_time()

        for audio_model in audio_models:
            audio_start = CombineProcessor.__parse_time(str(audio_model.start_time()))
            audio_end = CombineProcessor.__parse_time(str(audio_model.end_time()))
            overlap_start = max(audio_start, video_start)
            overlap_end = min(audio_end, video_end)

            if overlap_start < overlap_end:
                overlapping_models.append(audio_model)

        return overlapping_models

    def __validate_and_adjust_description(self, audio_model: AudioModel, video_model: VideoModel, run_manager: Optional[CallbackManagerForChainRun] = None) -> str:
        conversation = LLMChain(
            llm=self.llm,
            prompt=VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT,
            verbose=True,
            callbacks=run_manager.get_child() if run_manager else None
        )
        # Get response from OpenAI using LLMChain
        response: Dict[str, str] = conversation({"limit": self.__CHAR_LIMIT, "subtitle": audio_model.subtitle_text()})

        # Take out the Result: part of the response
        return response["text"].replace("Result:", "").strip()


# Now, caption_models contains your finalized caption models
    # def __init__(self, llm, verbose):
    #     self.llm = llm
    #     self.verbose = verbose

    # def process(self, audio_models, video_models) -> list:
    #     from langchain.chains import LLMChain

    #     def validate_and_adjust_description(
    #         audio_model: AudioModel, video_model: VideoModel
    #     ):
    #         validation = LLMChain(
    #             llm=self.llm, prompt=VIDEO_AUDIO_VALIDATING_PROMPT, verbose=self.verbose
    #         )
    #         response = validation(
    #             {
    #                 "subtitle": audio_model.subtitle_text,
    #                 "image_description": video_model.image_description,
    #             }
    #         )
    #         return response["text"].replace("Result:", "").strip()

    #     def milliseconds_to_srt_time(ms):
    #         if isinstance(ms, str) and "," in ms:
    #             return ms

    #         """Converts milliseconds to SRT time format 'HH:MM:SS,mmm'."""
    #         hours = int(ms // 3600000)
    #         minutes = int((ms % 3600000) // 60000)
    #         seconds = int((ms % 60000) // 1000)
    #         milliseconds = int(ms % 1000)

    #         return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    #     def find_overlapping_audio_models(
    #         video_model: VideoModel, audio_models: list[AudioModel]
    #     ):
    #         overlapping_models = []
    #         video_start = video_model.start_time
    #         video_end = video_model.end_time

    #         for audio_model in audio_models:
    #             audio_start = self.parse_time(str(audio_model.start_time))
    #             audio_end = self.parse_time(str(audio_model.end_time))
    #             overlap_start = max(audio_start, video_start)
    #             overlap_end = min(audio_end, video_end)

    #             if overlap_start < overlap_end:
    #                 overlapping_models.append(audio_model)

    #         return overlapping_models

    #     caption_models = []

    #     for video_model in video_models:
    #         overlapping_audio_models = find_overlapping_audio_models(
    #             video_model, audio_models
    #         )

    #         if overlapping_audio_models:
    #             # Create separate captions for each overlapping audio model
    #             last_end_time = video_model.start_time  # Initialize last_end_time

    #             for audio_model in overlapping_audio_models:
    #                 overlap_start = max(
    #                     video_model.start_time,
    #                     self.parse_time(audio_model.start_time),
    #                 )
    #                 overlap_end = min(
    #                     video_model.end_time,
    #                     self.parse_time(audio_model.end_time),
    #                 )

    #                 # Create a caption for the overlapping period
    #                 caption_text = f"[{validate_and_adjust_description(audio_model, video_model)}] {audio_model.subtitle_text}"
    #                 caption_model = CaptionModel(
    #                     milliseconds_to_srt_time(overlap_start),
    #                     milliseconds_to_srt_time(overlap_end),
    #                     caption_text,
    #                 )
    #                 caption_models.append(caption_model)

    #                 last_end_time = overlap_end  # Update last_end_time

    #             # If there is a gap after the last overlapping period, create a caption for that gap
    #             if last_end_time < video_model.end_time:
    #                 gap_start = last_end_time
    #                 gap_end = video_model.end_time
    #                 gap_caption_text = f"[{video_model.image_description}]"
    #                 gap_caption_model = CaptionModel(
    #                     milliseconds_to_srt_time(gap_start),
    #                     milliseconds_to_srt_time(gap_end),
    #                     gap_caption_text,
    #                 )
    #                 caption_models.append(gap_caption_model)

    #         else:
    #             # No overlapping audio, use video model's description for the entire duration
    #             caption_text = f"[{video_model.image_description}]"
    #             caption_model = CaptionModel(
    #                 milliseconds_to_srt_time(video_model.start_time),
    #                 milliseconds_to_srt_time(video_model.end_time),
    #                 caption_text,
    #             )
    #             caption_models.append(caption_model)

    #     return caption_models

    # def parse_time(self, s):
    #     """Converts a time string into milliseconds."""
    #     h, m, s = map(float, s.replace(",", ".").split(":"))
    #     return int((h * 3600 + m * 60 + s) * 1000)
