class VideoModel:
    def __init__(self, start_time, end_time, image_description):
        self._start_time = start_time
        self._end_time = end_time
        self._image_description = image_description

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

    @property
    def image_description(self):
        return self._image_description

    @image_description.setter
    def image_description(self, value):
        self._image_description = value

    def __str__(self):
        return f"start_time: {self.start_time}, end_time: {self.end_time}, image_description: {self.image_description}"


class AudioModel:
    def __init__(self, start_time, end_time, subtitle_text):
        self._start_time = start_time
        self._end_time = end_time
        self._subtitle_text = subtitle_text

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

    @property
    def subtitle_text(self):
        return self._subtitle_text

    @subtitle_text.setter
    def subtitle_text(self, value):
        self._subtitle_text = value

    def __str__(self):
        return f"start_time: {self.start_time}, end_time: {self.end_time}, subtitle_text: {self.subtitle_text}"


class CaptionModel:
    def __init__(self, start_time, end_time, closed_caption):
        self._start_time = start_time
        self._end_time = end_time
        self._closed_caption = closed_caption

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

    @property
    def closed_caption(self):
        return self._closed_caption

    @closed_caption.setter
    def closed_caption(self, value):
        self._closed_caption = value

    def __str__(self):
        return f"start_time: {self.start_time}, end_time: {self.end_time}, closed_caption: {self.closed_caption}"
