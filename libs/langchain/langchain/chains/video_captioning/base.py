from typing import List, Optional, Dict, Any

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.video_captioning.audio_service import AudioProcessor
from langchain.chains.video_captioning.combine_service import CombineProcessor
from langchain.chains.video_captioning.image_service import ImageProcessor
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel


class VideoCaptioningChain(Chain):
    llm: BaseLanguageModel
    prompt: PromptTemplate
    audio_processor: Optional[AudioProcessor] = None
    image_processor: Optional[ImageProcessor] = None
    combine_processor: Optional[CombineProcessor] = None

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt: PromptTemplate,
        verbose: bool = True,
        audio_processor: Optional[AudioProcessor] = None,
        image_processor: Optional[ImageProcessor] = None,
        combine_processor: Optional[CombineProcessor] = None,
        **kwargs,
    ):
        super().__init__(llm=llm, prompt=prompt, verbose=verbose, **kwargs)
        self.llm = llm
        self.prompt = prompt
        self.verbose = verbose
        self.audio_processor = audio_processor or AudioProcessor()
        self.image_processor = image_processor or ImageProcessor(
            llm=llm, verbose=verbose
        )
        self.combine_processor = combine_processor or CombineProcessor(
            llm=llm, verbose=verbose
        )

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
        audio_models = self.audio_processor.process(video_file_path)
        video_models = self.image_processor.process(video_file_path)
        caption_models = self.combine_processor.process(audio_models, video_models)
        srt_content = self.generate_srt_content(caption_models)
        return {"srt": srt_content}

    def format_srt_entry(self, index, caption_model):
        """Formats a single caption model into an SRT entry."""
        start_time = caption_model.start_time
        end_time = caption_model.end_time
        text = caption_model.closed_caption

        return f"{index}\n{start_time} --> {end_time}\n{text}\n"

    def generate_srt_content(self, caption_models: List[Dict[str, str]]) -> str:
        """Generates the full SRT content from a list of caption models."""
        srt_entries = []
        for index, model in enumerate(caption_models, start=1):
            srt_entry = self.format_srt_entry(index, model)
            srt_entries.append(srt_entry)

        return "\n".join(srt_entries)

    @property
    def _chain_type(self) -> str:
        return "video_captioning_chain"
