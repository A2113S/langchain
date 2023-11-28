from typing import List, Optional, Dict, Any

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.video_captioning.services.audio_service import AudioProcessor
from langchain.chains.video_captioning.services.caption_service import CaptionProcessor
from langchain.chains.video_captioning.services.combine_service import CombineProcessor
from langchain.chains.video_captioning.services.image_service import ImageProcessor
from langchain.chains.video_captioning.services.srt_service import SRTProcessor
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel


class VideoCaptioningChain(Chain):
    """
    Video Captioning Chain.
    """
    llm: BaseLanguageModel
    assemblyai_key: str
    prompt: Optional[PromptTemplate] = None
    verbose: Optional[bool] = True
    image_delta_threshold: Optional[bool] = 30
    closed_caption_char_limit: Optional[bool] = 20
    closed_caption_similarity_threshold: Optional[int] = 80
    use_unclustered_video_models: Optional[bool] = False

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["video_file_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["srt"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        if "video_file_path" not in inputs:
            raise ValueError(
                "Missing 'video_file_path' in inputs for video captioning."
            )
        video_file_path = inputs["video_file_path"]


        audio_processor = AudioProcessor(api_key = self.assemblyai_key)
        image_processor = ImageProcessor(threshold = self.image_delta_threshold)
        caption_processor = CaptionProcessor(
                llm=self.llm, verbose=self.verbose,
                similarity_threshold=self.closed_caption_similarity_threshold,
                use_unclustered_models=self.use_unclustered_video_models
            )
        combine_processor = CombineProcessor(
                llm=self.llm, verbose=self.verbose, char_limit=self.closed_caption_char_limit
            )
        srt_processor = SRTProcessor()
        # Get models for speech to text subtitles
        audio_models = audio_processor.process(video_file_path)

        # Get models for image frame description
        image_models = image_processor.process(video_file_path)

        # Get models for video event closed-captions
        video_models = caption_processor.process(image_models, run_manager)

        # Combine the subtitle models with the closed-caption models
        caption_models = combine_processor.process(audio_models, video_models, run_manager)

        # Convert the combined model to SRT format
        srt_content = srt_processor.process(caption_models)

        if run_manager:
            run_manager.on_text("Finished generating srt file from video file.")

        return {"srt": srt_content}

    @property
    def _chain_type(self) -> str:
        return "video_captioning_chain"
