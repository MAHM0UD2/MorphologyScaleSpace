import numpy as np
from PIL import Image as PILImage

class Image:
    def __init__(self, data):
        # Store the data as a NumPy array for fast math operations
        self._data = np.array(data)

    @property
    def height(self):
        # The number of rows
        return self._data.shape[0]

    @property
    def width(self):
        # The number of columns
        return self._data.shape[1]

    def __getitem__(self, indices):
        # Allows reading pixels using image[i, j]
        i, j = indices
        return self._data[i, j]

    def __setitem__(self, indices, value):
        # Allows modifying pixels using image[i, j] = value
        i, j = indices
        self._data[i, j] = value

    def as_array(self):
        # Helper to return the raw matrix when you need to do full-image math
        return self._data

    def __sub__(self, other: Image):
        arr_self = self.as_array()
        arr_other = other.as_array()

        return Image(arr_self - arr_other)

    def __add__(self, other: Image):
        arr_self = self.as_array()
        arr_other = other.as_array()

        return Image(arr_self + arr_other)

    def save_bmp(self, filepath):
        raw_array = self.as_array()
        visual_array = np.clip(raw_array, 0, 255).astype(np.uint8)
        PILImage.fromarray(visual_array).save(filepath)

    def absolute_log_difference(self, other: Image) -> Image:
        # old = |self - other|
        # new_value = c * ln(1+old)
        arr_self = self.as_array()
        arr_other = other.as_array()

        abs_error = np.abs(arr_self - arr_other)

        c = 255 / np.log1p(255)
        log_error = c * np.log1p(abs_error)

        discrete_log_error = np.round(log_error)

        return Image(discrete_log_error)