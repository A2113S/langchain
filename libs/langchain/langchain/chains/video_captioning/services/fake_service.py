from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.video_captioning.services.service import Processor


class FakeProcessor(Processor):
    """
    Fake processor for testing purposes.
    """
    def __init__(self, data):
        self.data = data

    def process(self, **kwargs) -> list:
        return self.data
    
