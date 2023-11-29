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
            for audio_model in audio_models:
                overlap_start, overlap_end = self._check_overlap(video_model, audio_model)
                if overlap_start:
                    if video_model.start_time < overlap_start:
                        caption_models.append(CaptionModel(video_model.start_time, overlap_start, video_model.image_description))
                    elif audio_model.start_time < overlap_start:
                        caption_models.append(CaptionModel(audio_model.start_time, overlap_start, audio_model.subtitle_text))

                    caption_text = f"[{self._validate_and_adjust_description(audio_model, video_model, run_manager)}] {audio_model.subtitle_text}"
                    caption_model = CaptionModel(overlap_start, overlap_end, caption_text)
                    caption_models.append(caption_model)

                    if video_model.end_time > overlap_end:
                        caption_models.append(CaptionModel(overlap_end, video_model.end_time, video_model.image_description))
                    elif audio_model.end_time > overlap_end:
                        caption_models.append(CaptionModel(overlap_end, audio_model.end_time, audio_model.subtitle_text))

        return caption_models

    @staticmethod
    def _check_overlap(video_model: VideoModel, audio_model: AudioModel):
        overlap_start = max(audio_model.start_time, video_model.start_time)
        overlap_end = min(audio_model.start_time, video_model.end_time)
        if overlap_start < overlap_end:
            return overlap_start, overlap_end
        return None, None

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
    