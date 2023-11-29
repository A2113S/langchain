from typing import Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.chains.video_captioning.models import (
    AudioModel,
    VideoModel,
    CaptionModel,
)
from langchain.chains.video_captioning.prompts import VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT
from langchain.chains.video_captioning.services.service import Processor
from langchain.schema.language_model import BaseLanguageModel


class CombineProcessor(Processor):
    def __init__(self, llm: BaseLanguageModel, verbose = True, char_limit: int = 20):
        self.llm = llm
        self.verbose = verbose

        # Adjust as needed. Be careful adjusting it too low because OpenAI may produce unwanted output
        self._CHAR_LIMIT = char_limit
    
    
    def process(self, video_models: List[VideoModel], audio_models: List[AudioModel], run_manager: Optional[CallbackManagerForChainRun] = None) -> List[CaptionModel]:
        caption_models = []
        
        for video_model in video_models:
            overlapping_audio_models = self._find_overlapping_audio_models(video_model, audio_models)

            run_manager.on_text(str(video_model.start_time) + ", " + str(video_model.end_time) + "\n") if run_manager else None

            if overlapping_audio_models:
                # Create separate captions for each overlapping audio model
                last_end_time = video_model.start_time  # Initialize last_end_time

                for audio_model in overlapping_audio_models:
                    overlap_start = max(video_model.start_time, audio_model.start_time)
                    overlap_end = min(video_model.end_time, audio_model.end_time)

                    # Create a caption for the overlapping period
                    caption_text = f"[{self._validate_and_adjust_description(audio_model, video_model, run_manager)}] {audio_model.subtitle_text}"
                    caption_model = CaptionModel(overlap_start, overlap_end, caption_text)
                    caption_models.append(caption_model)

                    last_end_time = overlap_end  # Update last_end_time

                # If there is a gap after the last overlapping period, create a caption for that gap
                if last_end_time < video_model.end_time:
                    gap_start = last_end_time
                    gap_end = video_model.end_time
                    gap_caption_text = f"[{video_model.image_description}]"
                    gap_caption_model = CaptionModel(gap_start, gap_end, gap_caption_text)
                    caption_models.append(gap_caption_model)

            else:
                # No overlapping audio, use video model's description for the entire duration
                caption_text = f"[{video_model.image_description}]"
                caption_model = CaptionModel(video_model.start_time, video_model.end_time, caption_text)
                caption_models.append(caption_model)

        return caption_models

    def _find_overlapping_audio_models(self, video_model: VideoModel, audio_models: List[AudioModel]) -> List[AudioModel]:
        overlapping_models = []
        video_start = video_model.start_time
        video_end = video_model.end_time

        for audio_model in audio_models:
            audio_start = audio_model.start_time
            audio_end = audio_model.end_time
            overlap_start = max(audio_start, video_start)
            overlap_end = min(audio_end, video_end)

            if overlap_start < overlap_end:
                overlapping_models.append(audio_model)

        return overlapping_models

    def _validate_and_adjust_description(self, audio_model: AudioModel, video_model: VideoModel, run_manager: Optional[CallbackManagerForChainRun] = None) -> str:
        conversation = LLMChain(
            llm=self.llm,
            prompt=VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT,
            verbose=True,
            callbacks=run_manager.get_child() if run_manager else None
        )
        # Get response from OpenAI using LLMChain
        response: Dict[str, str] = conversation({"limit": self._CHAR_LIMIT, "subtitle": audio_model.subtitle_text, "description": video_model.image_description})

        # Take out the Result: part of the response
        return response["text"].replace("Result:", "").strip()
    