    
from typing import Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.video_captioning.models import CaptionModel
from langchain.chains.video_captioning.services.service import Processor

class SRTProcessor(Processor):
    @staticmethod
    def process(caption_models: List[CaptionModel], run_manager: Optional[CallbackManagerForChainRun] = None) -> str:
        """Generates the full SRT content from a list of caption models."""
        srt_entries = []
        for index, model in enumerate(caption_models, start=1):
            srt_entries.append(model.to_srt_entry(index))

        return "\n".join(srt_entries)