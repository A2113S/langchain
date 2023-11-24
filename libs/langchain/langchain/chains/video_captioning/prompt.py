# flake8: noqa
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

video_caption_prompt = """You are caption organizer. If I give you caption, you have to organize the caption
                Organize mean:
                caption is raw closed caption that is not organized or not summarized or remove duplicate
make closed caption more efficient that if there is duplicate content, shrink to one and increase the end_time (so combine to one)
Make Better Closed caption too, understand before and after frame's closed caption and understand whats going on this video then make better.

                IMPORTANT: give only caption without description.
                """

VIDEO_OPTIMIZING_PROMPT = ChatPromptTemplate(
    messages=[
        SystemMessage(content=video_caption_prompt),
        HumanMessagePromptTemplate.from_template("{closed_caption}"),
    ]
)

task_description = """
As a caption organizer, your role is to evaluate and optimize video captions. Here's your task:
- Analyze the given subtitle from an audio track and the image description from a corresponding video frame.
- Determine if the subtitle and image description logically align.
- If they align, the image description is suitable as a closed caption.
- If they don't align, creatively adjust the image description to fit the subtitle, keeping the main idea intact.
- Produce a concise and coherent closed caption.
- Remember, the final output should be formatted as 'Result: [closed caption]' and should exclude the original subtitle.
Your goal is to ensure that the closed caption accurately represents the content and context of the video.
"""

VIDEO_AUDIO_VALIDATING_PROMPT = ChatPromptTemplate(
    messages=[
        SystemMessage(content=task_description),
        HumanMessagePromptTemplate.from_template(
            "Subtitle: {subtitle}, Image Description: {image_description}"
        ),
    ]
)
