# flake8: noqa
from langchain.prompts import PromptTemplate

prompt_template = """Generate closed captions for the video located at the following address: 
{address}"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["address"])

video_caption_prompt = """You are caption organizer. If I give you caption, you have to organize the caption
                Organize mean:
                caption is raw closed caption that is not organized or not summarized or remove duplicate
make closed caption more efficient that if there is duplicate content, shrink to one and increase the end_time (so combine to one)
Make Better Closed caption too, understand before and after frame's closed caption and understand whats going on this video then make better.

                IMPORTANT: give only caption without description.
                """

VIDEO_CAPTION_PROMPT = PromptTemplate.from_template(video_caption_prompt)
