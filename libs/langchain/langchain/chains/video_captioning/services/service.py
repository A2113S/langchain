from abc import ABC, abstractmethod
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
class Processor(ABC):
    @abstractmethod
    def process(self, run_manager: Optional[CallbackManagerForChainRun] = None):
        pass