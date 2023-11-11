# flake8: noqa
from langchain.prompts import PromptTemplate

prompt_template = """Generate closed captions for the video located at the following address: 
{address}"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["address"]
)