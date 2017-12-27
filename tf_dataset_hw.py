import tensorflow as tf
from dataset_hw import *


class HandWritingDatasetTF(HandWritingDataset):
    """
    Tensorflow extension to HandWritingDataset class.
    """
    def __init__(self, data_path, var_len_seq=None):
        super(HandWritingDatasetTF, self).__init__(data_path, var_len_seq)
        # Add tensorflow data types.
        self.sample_tf_type = [tf.int32, tf.float32, tf.float32]

class HandWritingDatasetConditionalTF(HandWritingDatasetConditional):
    """
    Tensorflow extension to HandWritingDataset class.
    """
    def __init__(self, data_path, var_len_seq=None, use_bow_labels=True):
        super(HandWritingDatasetConditionalTF, self).__init__(data_path, var_len_seq, use_bow_labels)
        # Add tensorflow data types.
        self.sample_tf_type = [tf.int32, tf.float32, tf.float32]


class HandWritingClassificationDataset(HandWritingClassificationDataset):
    """
    Tensorflow extension to HandWritingDatasetClassificationSOW class.
    """
    def __init__(self, data_path, var_len_seq=None, use_bow_labels=False, data_augmentation=False):
        super(HandWritingClassificationDataset, self).__init__(data_path, var_len_seq, use_bow_labels, data_augmentation)
        # Add tensorflow data types.
        self.sample_tf_type = [tf.int32, tf.float32, tf.float32]