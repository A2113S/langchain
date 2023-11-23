import cv2
import numpy as np
import os
from langchain.document_loaders import ImageCaptionLoader


class ImageProcessor:
    def process(self, video_file_path: str) -> list:
        frames = self.extract_frames(video_file_path)
        frame_transcriptions = [self.transcribe_image(frame) for frame in frames]
        return frame_transcriptions

    def extract_frames(self, video_file_path: str) -> list:
        capture = cv2.VideoCapture(video_file_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_duration = 1000 / fps
        frames = []

        ret, prev_frame = capture.read()
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ret else None

        while ret:
            ret, frame = capture.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.frame_difference_mse(prev_frame_gray, frame_gray):
                frames.append(frame)
                prev_frame_gray = frame_gray

        capture.release()
        return frames

    def transcribe_image(self, image) -> str:
        # Assuming 'image' is a file path, not an image array
        loader = ImageCaptionLoader(images=[image])
        list_docs = loader.load()
        return list_docs[0].page_content.replace("[SEP]", "").strip()

    def mse(self, imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def frame_difference_mse(self, prev_frame, curr_frame, threshold=1000):
        mse_value = self.mse(prev_frame, curr_frame)
        return mse_value > threshold
