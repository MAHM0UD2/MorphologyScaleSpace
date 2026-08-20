import numpy as np

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