import os

import cv2
import numpy as np

from langchain.chains.video_captioning.models import VideoModel
from langchain.document_loaders import ImageCaptionLoader
import transformers


transformers.logging.set_verbosity_error()


class ImageProcessor:
    def __init__(self, threshold: int = 30):
        self.threshold = threshold

    def process(self, video_file_path: str) -> list:
        return self.__extract_frames(video_file_path)
    
    def __extract_frames(self, video_file_path: str) -> list:

        # Define the path to the "output_frames" folder
        folder_path = 'test_data/output_frames'

        # Create the output directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        def __frame_difference(prev_frame, curr_frame):
            # Compute the absolute difference between the current frame and the previous frame
            diff = cv2.absdiff(prev_frame, curr_frame)
            # Thresholding to get the binary image, where white represents significant difference
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            # If there are any white pixels in thresh, the difference is significant
            return np.any(thresh)
        
        # Initialize the video capture
        capture = cv2.VideoCapture(video_file_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_duration = 1000 / fps

        video_models = []

        frameNr = 0
        ret, prev_frame = capture.read()
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ret else None
        prev_start_time = 0
        start_time = 0

        while ret:
            end_time = prev_start_time
            prev_start_time = capture.get(cv2.CAP_PROP_POS_MSEC)

            ret, frame = capture.read()
            if not ret:
                start_time = end_time + frame_duration
                break
            
            start_time = prev_start_time
            # Convert to grayscale for comparison
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compare with the previous frame
            if frameNr == 0 or __frame_difference(prev_frame_gray, frame_gray):
                end_time = capture.get(cv2.CAP_PROP_POS_MSEC)
                cv2.imwrite(f'{folder_path}/frame.jpg', frame)
                prev_frame_gray = frame_gray

                # List all .jpg files in the folder
                image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]

                # Create an instance of the ImageCaptionLoader
                loader = ImageCaptionLoader(images=image_files)

                # Load captions for the images
                list_docs = loader.load()

                video_model = VideoModel(start_time, end_time, list_docs[len(list_docs) - 1].page_content.replace("[SEP]", "").strip())
                video_models.append(video_model)

                frameNr += 1

        # Release the video capture object
        capture.release()
        return video_models

    # ------------------------------------- Existing code ------------------------------------- #

    # def __init__(self, llm, verbose):
    #     self.llm = llm
    #     self.verbose = verbose

    # def process(self, video_file_path: str) -> list:
    #     frames = self.extract_frames(video_file_path)
    #     video_models = [
    #         self.transcribe_image(frame, start_time, end_time)
    #         for (frame, start_time, end_time) in frames
    #     ]
    #     return self.optimize_caption(video_models)
    
    # def extract_frames(self, video_file_path: str) -> list:
    #     capture = cv2.VideoCapture(video_file_path)
    #     fps = capture.get(cv2.CAP_PROP_FPS)
    #     frame_duration = 1000 / fps
    #     frames = []
    #     prev_start_time = 0

    #     ret, prev_frame = capture.read()
    #     prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ret else None

    #     while ret:
    #         start_time = prev_start_time
    #         prev_start_time = capture.get(cv2.CAP_PROP_POS_MSEC)

    #         ret, frame = capture.read()
    #         if not ret:
    #             break

    #         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         if self.frame_difference_mse(prev_frame_gray, frame_gray):
    #             end_time = capture.get(cv2.CAP_PROP_POS_MSEC)
    #             frames.append((frame, start_time, end_time))
    #             prev_frame_gray = frame_gray

    #     capture.release()
    #     return frames

    # def transcribe_image(self, image, start_time, end_time) -> VideoModel:
    #     # Save frame image to file (assuming a write_frames_to_disk function)
    #     image_file_path = self.write_frames_to_disk(image)

    #     loader = ImageCaptionLoader(images=[image_file_path])
    #     list_docs = loader.load()
    #     description = list_docs[0].page_content.replace("[SEP]", "").strip()

    #     return VideoModel(start_time, end_time, description)

    # def optimize_caption(self, video_models) -> list:
    #     from langchain.chains import LLMChain

    #     optimizing = LLMChain(
    #         llm=self.llm, prompt=VIDEO_OPTIMIZING_PROMPT, verbose=self.verbose
    #     )

    #     caption_data = ""
    #     for model in video_models:
    #         caption_data += f"{str(model)}\n"

    #     result = optimizing({"closed_caption": caption_data})
    #     optimized_caption = result["text"]

    #     return self.text_to_models(optimized_caption)

    # def write_frames_to_disk(
    #     self, frame, output_dir="output_frames", file_prefix="frame"
    # ):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     # Generate a unique file name
    #     frame_id = len(os.listdir(output_dir))
    #     file_path = os.path.join(output_dir, f"{file_prefix}_{frame_id}.jpg")

    #     cv2.imwrite(file_path, frame)
    #     return file_path

    # def text_to_models(self, caption_data) -> list:
    #     video_models = []
    #     for line in caption_data.strip().split("\n"):
    #         if not line.strip():
    #             continue
    #         parts = line.split(",")
    #         start_time = float(parts[0].split(":")[1].strip())
    #         end_time = float(parts[1].split(":")[1].strip())
    #         image_description = parts[2].split(":")[1].strip()

    #         video_model = VideoModel(start_time, end_time, image_description)
    #         video_models.append(video_model)
    #     return video_models

    # def mse(self, imageA, imageB):
    #     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    #     err /= float(imageA.shape[0] * imageA.shape[1])
    #     return err

    # def frame_difference_mse(self, prev_frame, curr_frame, threshold=1000):
    #     mse_value = self.mse(prev_frame, curr_frame)
    #     return mse_value > threshold
