class VideoModel:
    def __init__(self, start_time, end_time, image_description):
        self.start_time = start_time
        self.end_time = end_time
        self.image_description = image_description

    def __str__(self):
        return f"start_time: {self.start_time}, end_time: {self.end_time}, image_description: {self.image_description}"

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_image_description(self):
        return self.image_description

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def set_image_description(self, image_description):
        self.image_description = image_description


class AudioModel:
    def __init__(self, start_time, end_time, subtitle_text):
        self.start_time = start_time
        self.end_time = end_time
        self.subtitle_text = subtitle_text

    def __str__(self):
        return f"start_time: {self.start_time}, end_time: {self.end_time}, subtitle_text: {self.subtitle_text}"

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_subtitle_text(self):
        return self.subtitle_text

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def set_subtitle_text(self, subtitle_text):
        self.subtitle_text = subtitle_text
