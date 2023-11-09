from typing import Any


class ImageProcessor:
    def process(self, video_file_path: str) -> str:
        frames = self.extract_frames(video_file_path)
        frame_transcriptions = [self.transcribe_image(frame) for frame in frames]
        return frame_transcriptions

    def extract_frames(self, video_file_path: str) -> str:
        # Logic to extract frames
        pass

    def transcribe_image(self, image: Any) -> str:
        # Logic to transcribe text from an image
        pass
