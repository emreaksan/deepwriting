import numpy as np

"""
Dataset class.

This class provides a basic interface to feed samples by using tensorflow's input pipeline (i.e., queues), hiding data
I/O latency bu using threads.

A `Dataset` object is given to `DataFeederTF` object which runs `sample_generator` method to enqueue the data queue.
The `sample_generator` method returns a generator yielding one sample at a time with shape and type specified by 
`sample_shape` and `sample_tf_type`.

The way the data is passed is not restricted. A child class can read the data from numpy array, list, dictionary, etc.
"""

class BaseDataset(object):
    """
    Acts as a data container. Loads and parses data, and provides basic functionality.
    """
    def __init__(self, data_path):
        self.data_dict = dict(np.load(data_path))

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value


    @property
    def sample_shape(self):
        """

        Returns:
            Shape of one sample (rank). A list of TensorShape objects, with the same length as self.sample_np_type and
            self.sample_tf_type.

        """
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, value):
        self._sample_shape = value

    @property
    def sample_np_type(self):
        """

        Returns:
            Numpy data type of one sample. It has the same structure with `self.sample_shape` and `self.sample_tf_type`.
            For example [np.int32, np.float32].

        """
        return self._sample_np_type

    @sample_np_type.setter
    def sample_np_type(self, value):
        self._sample_np_type = value

    @property
    def sample_tf_type(self):
        """
        Tensorflow counterpart of `sample_np_type`

        Returns:
            Tensorflow data type of one sample. It has the same structure with `self.sample_shape` and
            `self.sample_np_type`. For example [tf.int32, tf.float32].

        """
        return self._sample_tf_type

    @sample_tf_type.setter
    def sample_tf_type(self, value):
        self._sample_tf_type = value

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): that yields one sample consisting of a list of data elements.
        """
        raise NotImplementedError('Method is abstract.')