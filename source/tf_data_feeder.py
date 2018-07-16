import tensorflow as tf
import tensorflow.contrib.staging as tf_staging

import threading
from dataset import BaseDataset


class DataFeederTF(object):
    """
    Creates a tensorflow feeder in computational graph. The output variables are defined by the input dataset object.
    Uses threads to enqueue data asynchronously, and hides I/O latency.
    """

    def __init__(self, dataset, num_epochs, batch_size=16, queue_capacity=512, shuffle=True, allow_smaller_final_batch=False):
        """

        Args:
            dataset (Dataset):
            batch_size:
            queue_capacity:
        """
        assert(isinstance(dataset, BaseDataset))

        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.epoch = 0
        self.allow_smaller_final_batch = allow_smaller_final_batch

        self.queue_placeholders = []
        self.num_data_variables = len(self.dataset.sample_shape)

        for i in range(self.num_data_variables):
            self.queue_placeholders.append(tf.placeholder(self.dataset.sample_tf_type[i], shape=self.dataset.sample_shape[i]))

        if shuffle:
            self.input_queue = tf.RandomShuffleQueue(queue_capacity, min_after_dequeue=int(queue_capacity/2),
                                                     dtypes=self.dataset.sample_tf_type)
        else:
            self.input_queue = tf.FIFOQueue(queue_capacity, dtypes=self.dataset.sample_tf_type)

        self.enqueue_op = self.input_queue.enqueue(self.queue_placeholders)
        self.dequeue_op = self.input_queue.dequeue()

    def batch_queue(self, dynamic_pad=True, queue_capacity=512, queue_threads=4, name="batch_generator"):
        """
        A plain feeder is used and range of sequence lengths in a batch will be arbitrary.

        Args:
            dynamic_pad:
            queue_capacity:
            queue_threads:

        Returns:

        """
        self.batch = tf.train.batch(self.dequeue_op,
                                    batch_size=self.batch_size,
                                    capacity=int(queue_capacity / 2) + (queue_threads + 2) * self.batch_size,
                                    num_threads=queue_threads,
                                    shapes=self.dataset.sample_shape,
                                    enqueue_many=False,
                                    dynamic_pad=dynamic_pad,
                                    allow_smaller_final_batch=self.allow_smaller_final_batch,
                                    name=name)
        return self.batch

    def batch_queue_bucket(self, buckets, dynamic_pad=True, queue_capacity=128, queue_threads=4, name="batch_generator_bucket"):
        """
        Samples are first bucketed with respect to the sequence length. In this case the first entry of each sample in
        the dataset must be the sequence length.

        Args:
            buckets (list): a list of bucket boundaries (i.e., the edges of the buckets to use when bucketing samples)
            dynamic_pad:
            queue_capacity:
            queue_threads:

        Returns:

        """
        batch_seq_lens, self.batch = tf.contrib.training.bucket_by_sequence_length(
                                    input_length=self.dequeue_op[0],
                                    tensors=self.dequeue_op,
                                    batch_size=self.batch_size,
                                    bucket_boundaries=buckets,
                                    num_threads=queue_threads,
                                    capacity=queue_capacity,
                                    bucket_capacities=None,
                                    shapes=self.dataset.sample_shape,
                                    dynamic_pad=dynamic_pad,
                                    allow_smaller_final_batch=False,
                                    name=name)
        return self.batch

    def __enqueue(self, tf_session, tf_coord):
        """
        while (self.epoch < self.num_epochs) and (not self.terminated):
            self.epoch += 1
            sample_generator = self.dataset.sample_generator()
            for sample in sample_generator:
                feed_dict = {pl:val for pl, val in zip(self.queue_placeholders, sample)}
                tf_session.run(self.enqueue_op, feed_dict=feed_dict)
        """
        sample_generator = self.dataset.sample_generator()
        while not tf_coord.should_stop():
            try:
                sample = next(sample_generator)
                feed_dict = {pl: val for pl, val in zip(self.queue_placeholders, sample)}
                tf_session.run(self.enqueue_op, feed_dict=feed_dict)
            except StopIteration:
                sample_generator = self.dataset.sample_generator()
            except tf.errors.CancelledError:
                pass

    def init(self, tf_session, tf_coord):
        # TODO: it is not multi-threaded.
        self.enqueue_threads = threading.Thread(target=self.__enqueue, args=[tf_session, tf_coord])
        self.enqueue_threads.start()


class TFStagingArea(object):

    def __init__(self, tensors, device_name=None):
        if device_name is None:
            self._staging_area = self._create_staging_area(tensors)
        else:
            with tf.device(device_name):
                self._staging_area = self._create_staging_area(tensors)
        self._preload_op = self._staging_area.put(tensors)
        self._tensors = self._staging_area.get()

    def _create_staging_area(self, tensors):
        return tf_staging.StagingArea(dtypes=[tensor.dtype for tensor in tensors], shapes=[tensor.shape for tensor in tensors])

    @property
    def preload_op(self):
        return self._preload_op

    @property
    def tensors(self):
        return self._tensors
