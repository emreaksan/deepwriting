import numpy as np
import scipy.signal
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import utils_hw as utils
from dataset import BaseDataset


class HandWritingDataset(BaseDataset):
    """
    Customized for handwriting dataset.

    Stroke data is assumed to be consisting of 3 dimensions x, y and pen, respectively. If the stroke data is required
    to be concatenated with other modalities, then stroke data relies in the first 3 dimensions.

    Args:
        data_path (str): path to numpy dataset file. See data_scripts/preprocessing.py for details.
        var_len_seq (bool): whether the dataset consists of variable-length sequences or not. If set to False, then
            it is determined from the dataset samples.
    """
    def __init__(self, data_path, var_len_seq=False):
        super(HandWritingDataset, self).__init__(data_path)

        # TODO new_dataset
        #self.samples = self.data_dict['strokes']
        self.samples = self.data_dict['samples'] if 'samples' in self.data_dict.keys() else self.data_dict['strokes']
        self.char_labels = self.data_dict['char_labels']
        self.subject_labels = self.data_dict['subject_labels']
        self.texts= self.data_dict['texts']

        self.feature_size = self.samples[0].shape[-1] # x,y,pen

        # Models require input and target dimensionality. They are useful if the inputs and targets are concatenation
        # of different modalities. They are used to split the input/target into components.
        self.input_dims = [self.feature_size]
        self.target_dims = [2, 1] # Stroke, pen

        # The dimensions with None will be padded if seq_len isn't passed.
        self.sequence_length = None if var_len_seq else self.extract_seq_len()
        self.is_dynamic = self.sequence_length == None

        # sequence length, strokes, targets (i.e., strokes).
        self.sample_shape = [[], [self.sequence_length, self.feature_size], [self.sequence_length, self.feature_size]]
        self.sample_np_type = [np.int32, np.float32, np.float32]

        self.num_samples = len(self.samples)

        # Preprocessing
        self.normalization = 'normalization' in self.data_dict['preprocessing']
        if not self.normalization:
            print("Warning: data is not normalized.")
        elif not ('mean' in self.data_dict):
            raise Exception("Normalization statistics (mean and std) are missing.")
        else:
            self.norm_mean = self.data_dict['mean']
            self.norm_std = self.data_dict['std']

        self.relative_representation = 'relative_representation' in self.data_dict['preprocessing']
        self.offset_removal = 'origin_translation' in self.data_dict['preprocessing']

        self.scale = 'scale' in self.data_dict['preprocessing']
        if self.scale and not('min' in self.data_dict):
            pass
            #raise Exception("Scaling statistics (min and max) are missing.")
        else:
            self.scale_min = self.data_dict['min']
            self.scale_max = self.data_dict['max']

    def preprocess_raw_sample(self, sample):
        """
        Gets a raw (!) sample and applies preprocessing steps that the dataset has been applied.

        Args:
            sample: [seq_len, 3]

        Returns:

        """
        sample_copy = np.copy(sample[:, :3])
        statistics = {}
        if self.scale:
            sample_copy[:, [0, 1]] = ((sample-self.scale_min)/(self.scale_max-self.scale_min))[:, [0, 1]]
        if self.offset_removal:
            statistics['x_offset'] = sample_copy[0, 0]
            statistics['y_offset'] = sample_copy[0, 1]
            sample_copy[:, 0] -= statistics['x_offset']
            sample_copy[:, 1] -= statistics['y_offset']
        if self.relative_representation:
            source = np.vstack((sample_copy[0], sample_copy))
            sample_copy = np.diff(source, axis=0)
            sample_copy[:, 2] = sample[:, 2]  # Keep original pen information since it is already relative.
        if self.normalization:
            sample_copy[:, [0, 1]] = ((sample_copy-self.norm_mean)/self.norm_std)[:, [0, 1]]

        return sample_copy, statistics

    def undo_preprocess(self, sample, statistics=None):
        """
        Applies preprocessing in reverse order by using statistics parameters.

        Args:
            sample (numpy.ndarray): [seq_len, 3]
            statistics (dict): Contains dataset ("min", "max", "mean", "std") and sample ("x_offset", "y_offset")
                statistics. If a (dataset statistics) key is not found in the dictionary or has None value, then class
                statistics will be used.

        Returns:
            (numpy.ndarray): [seq_len, 3]
        """
        if statistics is None:
            statistics = {}

        sample_copy = np.copy(sample[:, :3])
        if self.normalization:
            mean_ = self.norm_mean
            std_ = self.norm_std
            if ('mean' in statistics) and (statistics['mean'] is not None):
                mean_ = statistics['mean']
                std_ = statistics['std']
            sample_copy[:, :2] = (sample_copy*std_ + mean_)[:, :2]

        if self.relative_representation:
            sample_copy = np.cumsum(sample_copy, 0)  # Assuming that the sequence always starts with 0.

        if self.offset_removal and 'x_offset' in statistics:
            sample_copy[:, 0] += statistics['x_offset']
            sample_copy[:, 1] += statistics['y_offset']

        if self.scale:
            min_ = self.scale_min
            max_ = self.scale_max
            if ('min' in statistics) and (statistics['min'] is not None):
                min_ = statistics['min']
                max_ = statistics['max']
            sample_copy[:, :2] = (sample_copy[:,:3]*(max_-min_) + min_)[:, :2]

        sample_copy[:, 2] = sample[:, 2]

        return sample_copy


    def prepare_for_visualization(self, sample, detrend_sample=False):
        """
        TODO: Move this method into a more proper class.

        Args:
            sample:

        Returns:

        """
        sample_copy = np.copy(sample[:,:3])
        if self.normalization:
            sample_copy = sample_copy*self.norm_std+self.norm_mean
        if detrend_sample:
            sample_copy[:,1] = scipy.signal.detrend(sample_copy[:,1])
        if self.relative_representation:
            sample_copy = np.cumsum(sample_copy, 0) # Assuming that the sequence always starts with 0.

        sample_copy[:,2] = sample[:,2]

        return sample_copy

    def undo_normalization(self, sample, detrend_sample=False):
        """
        TODO: Move this method into a more proper class.

        Args:
            sample:

        Returns:

        """
        sample_copy = np.copy(sample[:,:3])
        if self.normalization:
            sample_copy = sample_copy*self.norm_std+self.norm_mean
        if detrend_sample:
            sample_copy[:,1] = scipy.signal.detrend(sample_copy[:,1])
        sample_copy[:,2] = sample[:,2]

        return sample_copy

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): each sample is a list of data elements.
        """
        for stroke in self.samples:
            yield [stroke.shape[0], stroke, stroke]

    def fetch_sample(self, sample_idx):
        """
        Prepares one data sample (i.e. return of sample_generator) given index.

        Args:
            sample_idx:

        Returns:

        """
        stroke = self.samples[sample_idx]
        return [stroke.shape[0], stroke, stroke]

    # TODO Auxiliary methods can be in utils.
    def get_seq_len_histogram(self, num_bins=10, collapse_first_and_last_bins=[1, -1]):
        """
        Creates a histogram of sequence-length.
        Args:
            num_bins:
            collapse_first_and_last_bins: selects bin edges between the provided indices by discarding from the
            first
                and last bins.

        Returns:
            (list): bin edges.
        """
        seq_lens = [s.shape[0] for s in self.samples]
        h, bins = np.histogram(seq_lens, bins=num_bins)
        if collapse_first_and_last_bins is not None:
            return [int(b) for b in bins[collapse_first_and_last_bins[0]:collapse_first_and_last_bins[1]]]
        else:
            return [int(b) for b in bins]

    def extract_seq_len(self):
        seq_lens = [s.shape[0] for s in self.samples]

        if max(seq_lens) == min(seq_lens):
            return min(seq_lens)
        else:
            return None


class HandWritingDatasetConditional(HandWritingDataset):
    """
    Uses character labels.

    In contrast to HandWritingDataset dataset (i.e., non-conditional), concatenates one-hot-vector char labels with
    strokes.

    Args:
        data_path (str): path to numpy dataset file. See data_scripts/preprocessing.py for details.
        var_len_seq (bool): whether the dataset consists of variable-length sequences or not. If set to False, then
            it is determined from the dataset samples.
        use_bow_labels (bool): whether beginning-of-word labels (bow_labels) are yielded as model inputs or not.
    """
    def __init__(self, data_path, var_len_seq=None, use_bow_labels=True):
        super(HandWritingDatasetConditional, self).__init__(data_path, var_len_seq)

        self.use_bow_labels = use_bow_labels

        if not('alphabet' in self.data_dict):
            raise Exception("Alphabet is missing.")

        self.alphabet = self.data_dict['alphabet']
        self.alphabet_size = len(self.alphabet)

        self.feature_size = self.samples[0].shape[-1]  # x,y,pen
        # Models require input and target dimensionality. They are useful if the inputs and targets are concatenation
        # of different modalities. They are used to split the input/target into components.
        self.input_dims = [self.feature_size, len(self.alphabet)]
        self.target_dims = [2, 1, len(self.alphabet), 1]  # Stroke, pen, character labels, eoc
        if use_bow_labels:
            self.input_dims = [self.feature_size, len(self.alphabet), 1]
            self.target_dims = [2, 1, len(self.alphabet), 1, 1]  # Stroke, pen, character labels, eoc, bow

        int_alphabet = np.expand_dims(np.array(range(self.alphabet_size)), axis=1)

        self.char_encoder = LabelEncoder()
        self.char_encoder.fit(self.alphabet)
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.one_hot_encoder.fit(int_alphabet)
        self.__encode_labels()

        self.eoc_labels = self.data_dict['eoc_labels']
        self.boc_labels = self.data_dict['boc_labels'] if 'boc_labels' in self.data_dict.keys() else self.data_dict['soc_labels']
        self.eow_labels = self.data_dict['eow_labels']
        self.bow_labels = self.data_dict['bow_labels'] if 'bow_labels' in self.data_dict.keys() else self.data_dict['sow_labels']

        # sequence length, strokes, targets (i.e., strokes+end-of-character).
        # The dimensions with None will be padded if seq_len isn't passed.
        self.sample_shape = [[], [self.sequence_length, sum(self.input_dims)],  [self.sequence_length, sum(self.target_dims)]]

    def text_to_one_hot(self, text):
        integer_labels = self.char_encoder.transform(list(text))
        return self.one_hot_encoder.transform(np.expand_dims(integer_labels, axis=1))

    def int_labels_to_one_hot(self, int_labels):
        return self.one_hot_encoder.transform(np.expand_dims(int_labels, axis=1))

    def logit_to_one_hot(self, one_hot):
        integer_labels = np.argmax(one_hot, -1)
        return self.int_labels_to_one_hot(integer_labels)

    def one_hot_to_int_labels(self, one_hot):
        return np.argmax(one_hot, -1)

    def int_labels_to_text(self, int_labels):
        text_labels = utils.simplify_int_labels(int_labels)
        text = self.char_encoder.inverse_transform(text_labels)

        return text

    def __encode_labels(self):
        """
        Encodes integer character labels as one-hot vectors.

        Returns:

        """
        self.one_hot_char_labels = []
        for idx, label in enumerate(self.data_dict['char_labels']):
            self.one_hot_char_labels .append(self.one_hot_encoder.transform(np.expand_dims(label, axis=1)))

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): each sample is a list of data elements.
        """
        for stroke, char_label, eoc_label, bow_label in zip(self.samples, self.one_hot_char_labels, self.eoc_labels, self.bow_labels):
            bow_label_ = np.expand_dims(bow_label, axis=1)
            eoc_label_ = np.expand_dims(eoc_label, axis=1)
            if self.use_bow_labels:
                yield [stroke.shape[0], np.float32(np.hstack([stroke, char_label, bow_label_])), np.float32(np.hstack([stroke, char_label, eoc_label_, bow_label_]))]
            else:
                yield [stroke.shape[0], np.float32(np.hstack([stroke, char_label])), np.float32(np.hstack([stroke, char_label, eoc_label_]))]

    def fetch_sample(self, sample_idx):
        """
        Prepares one data sample (i.e. return of sample_generator) given index.

        Args:
            sample_idx:

        Returns:

        """
        stroke = self.samples[sample_idx]
        char_label = self.one_hot_char_labels[sample_idx]
        eoc_label = np.expand_dims(self.eoc_labels[sample_idx], axis=1)

        if self.use_bow_labels:
            bow_label = np.expand_dims(self.bow_labels[sample_idx], axis=1)
            return [stroke.shape[0], np.expand_dims(np.float32(np.hstack([stroke, char_label, bow_label])), axis=0), np.expand_dims(np.float32(np.hstack([stroke, char_label, eoc_label, bow_label])), axis=0)]
        else:
            return [stroke.shape[0], np.expand_dims(np.float32(np.hstack([stroke, char_label])), axis=0), np.expand_dims(np.float32(np.hstack([stroke, char_label, eoc_label])), axis=0)]


class HandWritingClassificationDataset(HandWritingDatasetConditional):
    """
    Handwriting dataset for character classification/segmentation models. In contrast to parent class
    HandWritingDatasetConditional, its sample_generator method yields only strokes as model input and
    [char_label, eoc_label, (bow_label)] as model target.

    Args:
        data_path (str): path to numpy dataset file. See data_scripts/preprocessing.py for details.
        var_len_seq (bool): whether the dataset consists of variable-length sequences or not. If set to False, then
            it is determined from the dataset samples.
        use_bow_labels (bool): whether beginning-of-word labels (bow_labels) are yielded as model targets or not.
        data_augmentation (bool): whether to apply data augmentation or not. If set True, strokes are scaled randomly.
    """

    def __init__(self, data_path, var_len_seq=None, use_bow_labels=False, data_augmentation=False):
        super(HandWritingClassificationDataset, self).__init__(data_path, var_len_seq)

        self.bow_target = use_bow_labels
        self.data_augmentation = data_augmentation

        self.input_dims = [self.samples[0].shape[-1]]
        self.feature_size = sum(self.input_dims)

        if self.bow_target:
            self.target_dims = [self.alphabet_size, 1, 1]  # char_labels, end-of-character, sow
        else:
            self.target_dims = [self.alphabet_size, 1] #char_labels, end-of-character

        # sequence length, strokes, targets
        # The dimensions with None will be padded if sequence_length isn't passed.
        self.sample_shape = [[], [self.sequence_length, sum(self.input_dims)], [self.sequence_length, sum(self.target_dims)]]


    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): each sample is a list of data elements.
        """
        if self.bow_target:
            for stroke, char_label, eoc_label, bow_label in zip(self.samples, self.one_hot_char_labels, self.eoc_labels, self.bow_labels):
                if self.data_augmentation:
                    stroke_augmented = stroke.copy()
                    stroke_augmented *= np.random.uniform(0.7,1.3, (1))
                else:
                    stroke_augmented = stroke
                yield [stroke.shape[0], stroke_augmented, np.float32(np.hstack([char_label, np.expand_dims(eoc_label,-1), np.expand_dims(bow_label,-1)]))]
        else:
            for stroke, char_label, eoc_label in zip(self.samples, self.one_hot_char_labels, self.eoc_labels):
                if self.data_augmentation:
                    stroke_augmented = stroke.copy()
                    stroke_augmented *= np.random.uniform(0.7,1.3, (1))
                else:
                    stroke_augmented = stroke
                yield [stroke.shape[0], stroke_augmented, np.float32(np.hstack([char_label, np.expand_dims(eoc_label,-1)]))]


    def fetch_sample(self, sample_idx):
        """
        Prepares one data sample (i.e. return of sample_generator) given index.

        Args:
            sample_idx:

        Returns:

        """
        stroke = np.expand_dims(self.samples[sample_idx], axis=0)
        char_label = self.one_hot_char_labels[sample_idx]
        eoc_label = np.expand_dims(self.eoc_labels[sample_idx], -1)
        bow_label = np.expand_dims(self.bow_labels[sample_idx], -1)

        if self.bow_target:
            return [stroke.shape[0], stroke, np.expand_dims(np.float32(np.hstack([char_label, eoc_label, bow_label])), axis=1)]
        else:
            return [stroke.shape[0], stroke, np.expand_dims(np.float32(np.hstack([char_label, eoc_label])), axis=1)]