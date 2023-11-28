class BaseModel:
    def __init__(self, start_time, end_time):
        self._start_time = start_time
        self._end_time = end_time

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    def __str__(self):
        return f"start_time: {self.start_time}, end_time: {self.end_time}"


class VideoModel(BaseModel):
    def __init__(self, start_time, end_time, image_description):
        super().__init__(start_time, end_time)
        self._image_description = image_description

    # Instantiate a video model from caption data which represents the description of what is currently happening in the video segment
    @staticmethod
    def convert(caption_data:str):
        line = caption_data.strip().split("\n")
        if not line.strip():
            raise ValueError("The 'caption_data' parameter cannot be empty.")

        parts = line.split(",")
        start_time = float(parts[0].split(":")[1].strip())
        end_time = float(parts[1].split(":")[1].strip())
        image_description = parts[2].split(":")[1].strip()

        return VideoModel(start_time, end_time, image_description)

    @property
    def image_description(self):
        return self._image_description

    @image_description.setter
    def image_description(self, value):
        self._image_description = value

    def __str__(self):
        return f"{super().__str__()}, image_description: {self.image_description}"
    
    def similarity_score(self, other):
        # Tokenize the image descriptions by extracting individual words, stripping trailing 's' (plural = singular)
        # and converting the words to lowercase in order to be case-insensitive
        self_tokenized = set(word.lower().rstrip('s') for word in self.image_description.split())
        other_tokenized = set(word.lower().rstrip('s') for word in other.image_description.split())

        # Find common words
        common_words = self_tokenized.intersection(other_tokenized)

        # Calculate similarity score
        similarity_score = len(common_words) / max(len(self_tokenized), len(other_tokenized)) * 100

        return similarity_score


class AudioModel(BaseModel):
    def __init__(self, start_time, end_time, subtitle_text):
        super().__init__(start_time, end_time)
        self._subtitle_text = subtitle_text

    @property
    def subtitle_text(self):
        return self._subtitle_text

    @subtitle_text.setter
    def subtitle_text(self, value):
        self._subtitle_text = value

    def __str__(self):
        return f"{super().__str__()}, subtitle_text: {self.subtitle_text}"


class CaptionModel(BaseModel):
    def __init__(self, start_time, end_time, closed_caption):
        super().__init__(start_time, end_time)
        self._closed_caption = closed_caption

    @property
    def closed_caption(self):
        return self._closed_caption

    @closed_caption.setter
    def closed_caption(self, value):
        self._closed_caption = value

    def __str__(self):
        return f"{super().__str__()}, closed_caption: {self.closed_caption}"

    def to_srt_entry(self, index):
        return f"{index}\n{self._start_time} --> {self._end_time}\n{self._closed_caption}\n"