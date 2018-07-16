import sys
import tf_loss
import tensorflow as tf
import numpy as np

from tf_models import VRNN, VRNNGMM
from tf_rnn_cells import *

"""
If opencv is installed, then you can visualize images of real and synthetic handwriting samples in tensorboard. 
Note that the experiment folder takes much more space (~2-3 GB). 
"""
from importlib import util as importlib_util
VISUAL_MODE = False
if importlib_util.find_spec("cv2") is not None:
    from visualize_hw import draw_stroke_cv2
    from utils_visualization import plot_and_get_image
    VISUAL_MODE = True
    print("VISUAL_MODE is active.")


class HandwritingVRNNModel(VRNN):
    def __init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, data_processor, batch_size=-1, mode="training"):
        VRNN.__init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=batch_size, mode=mode)

        self.dataset_obj = data_processor
        self.pen_loss_weight = self.config.get('loss_weights', {}).get('pen_loss', 1)

        # TODO: Create a dictionary just for cell arguments.
        self.vrnn_cell_args = config
        self.vrnn_cell_args['input_dims'] = self.input_dims

        # See `create_image_summary` method for details.
        self.img_summary_entries = []
        self.ops_img_summary = {}
        self.use_img_summary = self.config.get("img_summary_every_step", 0) > 0 and VISUAL_MODE

    def get_constructors(self):
        self.vrnn_cell_constructor = getattr(sys.modules[__name__], self.config['vrnn_cell_cls'])

    def build_predictions_layer(self):
        # Assign rnn outputs.
        self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, self.out_mu, self.out_sigma, self.out_rho, self.out_pen = self.outputs

        # For analysis.
        self.norm_p_mu = tf.norm(self.p_mu, axis=-1)
        self.norm_p_sigma = tf.norm(self.p_sigma, axis=-1)
        self.norm_q_mu = tf.norm(self.q_mu, axis=-1)
        self.norm_q_sigma = tf.norm(self.q_sigma, axis=-1)
        self.norm_out_mu = tf.norm(self.out_mu, axis=-1)
        self.norm_out_sigma = tf.norm(self.out_sigma, axis=-1)

        # TODO: Sampling option.
        self.output_sample = tf.concat([self.out_mu, tf.round(self.out_pen)], axis=2)
        self.input_sample = self.inputs
        self.output_dim = self.output_sample.shape.as_list()[-1]

        self.ops_evaluation['output_sample'] = self.output_sample
        self.ops_evaluation['p_mu'] = self.p_mu
        self.ops_evaluation['p_sigma'] = self.p_sigma
        self.ops_evaluation['q_mu'] = self.q_mu
        self.ops_evaluation['q_sigma'] = self.q_sigma
        self.ops_evaluation['state'] = self.output_state

        # In case we want to draw samples from output distribution instead of using mean.
        self.ops_evaluation['out_mu'] = self.out_mu
        self.ops_evaluation['out_sigma'] = self.out_sigma
        self.ops_evaluation['out_rho'] = self.out_rho
        self.ops_evaluation['out_pen'] = self.out_pen

        # Mask for precise loss calculation.
        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_length, maxlen=tf.reduce_max(self.input_seq_length), dtype=tf.float32), -1)

    def build_loss(self):
        if self.is_training or self.is_validation:
            targets_mu = self.target_pieces[0]
            targets_pen = self.target_pieces[1]

            if self.reconstruction_loss_key not in self.ops_loss:
                with tf.name_scope('reconstruction_loss'):
                    # Gaussian log likelihood loss (bivariate)
                    if self.reconstruction_loss == 'nll_normal_bi':
                        self.ops_loss[self.reconstruction_loss_key] = -self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_normal_bivariate(targets_mu, self.out_mu, self.out_sigma, self.out_rho, reduce_sum=False))
                    # Gaussian log likelihood loss (diagonal covariance)
                    elif self.reconstruction_loss == 'nll_normal_diag':
                        self.ops_loss[self.reconstruction_loss_key] = -self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_normal_diag_cov(targets_mu, self.out_mu, self.out_sigma, reduce_sum=False))
                    # L1 norm.
                    elif self.reconstruction_loss == "l1":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.absolute_difference(targets_mu, self.out_mu, reduction='none'))
                    # Mean-squared error.
                    elif self.reconstruction_loss == "mse":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.mean_squared_error(targets_mu, self.out_mu, reduction='none'))
                    else:
                        raise Exception("Undefined loss.")

            if "loss_pen" not in self.ops_loss:
                with tf.name_scope('pen_reconstruction_loss'):
                    # Bernoulli loss for pen information.
                    self.ops_loss['loss_pen'] = -self.pen_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_bernoulli(targets_pen, self.out_pen, reduce_sum=False))

            VRNN.build_loss(self)

    def create_image_summary(self, undo_preprocessing_func, img_stroke_shape=(1,120,1200,1), img_norm_shape=(1,800,1200,3)):
        """
        Creates placeholder and summary operations for image summaries. Supports two types of summaries:
        (1) stroke images.
        (2) image visualization of plots for a given sample. Note that this is required to visualize model performance
            on a test sample over training.

        In order to add a new type one should create an `img_entry` (see `stroke_img_entry` and `norm_plot_img_entry`)
        and register graph nodes as well as a post-processing function (see `post_processing_func` field).

        When `get_image_summary` method is called, for every registered `op` first evaluated results are converted into
        images and stored in containers (`container_img`). Then a summary object is created by passing these containers
        to tf.placeholders (`container_img_placeholders`).

        Args:
            session: tensorflow session.
            writer: summary writer.
            undo_preprocessing_func: function to undo normalization and preprocessing operations on model outputs.
            img_stroke_shape: shape of stroke images.
            img_norm_shape: shape of norm plot images.

        Returns:
        """
        # Post-processing functions for images.
        def norm_plot_img_func(img_data):
            return plot_and_get_image(img_data, axis_off=False)

        def stroke_img_func(img_data):
            return draw_stroke_cv2(undo_preprocessing_func(img_data), factor=1)

        if self.use_img_summary:
            # Make a separation between different types of images and provide corresponding functionality.
            stroke_img_entry = {}
            stroke_img_entry['img_shape'] = img_stroke_shape
            stroke_img_entry['num_img'] = img_stroke_shape[0]
            stroke_img_entry['data_type'] = tf.uint8
            stroke_img_entry['post_processing_func'] = stroke_img_func
            stroke_img_entry['ops'] = {}
            stroke_img_entry['ops']['stroke_output'] = self.output_sample
            if self.is_sampling is False:
                stroke_img_entry['ops']['stroke_input'] = self.input_sample

            norm_plot_img_entry = {}
            norm_plot_img_entry['img_shape'] = img_norm_shape
            norm_plot_img_entry['num_img'] = img_norm_shape[0]
            norm_plot_img_entry['data_type'] = tf.uint8
            norm_plot_img_entry['post_processing_func'] = norm_plot_img_func
            norm_plot_img_entry['ops'] = {}
            norm_plot_img_entry['ops']['norm_q_mu'] = self.norm_q_mu
            norm_plot_img_entry['ops']['norm_p_mu'] = self.norm_p_mu

            self.img_summary_entries.append(stroke_img_entry)
            self.img_summary_entries.append(norm_plot_img_entry)
            # Graph nodes to be evaluated by calling session.run
            self.ops_img_summary = {}
            # Create placeholders and containers for intermediate results.
            self.container_img = {}
            self.container_img_placeholders = {}
            self.container_img_feed_dict = {}

            for summary_dict in self.img_summary_entries:
                for op_name, summary_op in summary_dict['ops'].items():
                    self.ops_img_summary[op_name] = summary_op
                    # To store images.
                    self.container_img[op_name] = np.zeros(summary_dict['img_shape'])
                    # To pass images to summary
                    self.container_img_placeholders[op_name] = tf.placeholder(summary_dict['data_type'], summary_dict['img_shape'])
                    # Summary.
                    tf.summary.image(op_name, self.container_img_placeholders[op_name], collections=[self.mode+'_summary_img'], max_outputs=summary_dict['num_img'])
                    # Feed dictionary.
                    self.container_img_feed_dict[self.container_img_placeholders[op_name]] = 0

            self.img_summary = tf.summary.merge_all(self.mode+'_summary_img')

    def get_image_summary(self, session, ops_img_summary_evaluated=None, seq_len=500):
        """
        Evaluates the model, creates output images, plots and prepares a summary entry.

        Args:
            ops_img_summary_evaluated: list of summary inputs. If None passed, then the model is assumed to be in
            `sampling` mode.
            seq_len: length of a synthetic sample.

        Returns:
            summary entry for summary_writer.
        """
        if self.use_img_summary:
            if ops_img_summary_evaluated is None: # Inference mode.
                ops_img_summary_evaluated = self.sample_unbiased(session, seq_len=seq_len, ops_eval=self.ops_img_summary)[0]

            # Create images.
            for summary_dict in self.img_summary_entries:
                post_processing_func = summary_dict['post_processing_func']
                for op_name, summary_op in summary_dict['ops'].items():
                    for i in range(summary_dict['num_img']):
                        self.container_img[op_name][i] = np.float32(post_processing_func(ops_img_summary_evaluated[op_name][i]))
                    self.container_img_feed_dict[self.container_img_placeholders[op_name]] = self.container_img[op_name]

            img_summary = session.run(self.img_summary, self.container_img_feed_dict)

            return img_summary
        else:
            return None


class HandwritingVRNNGmmModel(VRNNGMM, HandwritingVRNNModel):
    def __init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, data_processor, batch_size=-1, mode="training"):
        VRNNGMM.__init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=batch_size, mode=mode)

        self.dataset_obj = data_processor
        self.text_to_label_fn = data_processor.text_to_one_hot

        self.pen_loss_weight = self.config.get('loss_weights', {}).get('pen_loss', 1)
        self.eoc_loss_weight = self.config.get('loss_weights', {}).get('eoc_loss', 1)
        self.bow_loss_weight = self.config.get('loss_weights', {}).get('bow_loss', None)

        self.use_bow_loss = False if self.bow_loss_weight is None else True
        self.use_bow_labels = config.get('use_bow_labels', True)

        # TODO: Create a dictionary just for cell arguments.
        self.vrnn_cell_args = config
        self.vrnn_cell_args['input_dims'] = self.input_dims

        if target_op is not None or self.is_training or self.is_validation:
            self.target_pieces = tf.split(self.targets, target_dims, axis=2)
            # TODO Swap pen and char targets. Parent `VRNNGMM` class expects class labels as the second entry.
            tmp_targets_pen = self.target_pieces[1]
            self.target_pieces[1] = self.target_pieces[2]
            self.target_pieces[2] = tmp_targets_pen

        # See `create_image_summary` method for details.
        self.img_summary_entries = []
        self.ops_img_summary = {}
        self.use_img_summary = self.config.get("img_summary_every_step", 0) > 0 and VISUAL_MODE

    def get_constructors(self):
        self.vrnn_cell_constructor = getattr(sys.modules[__name__], self.config['vrnn_cell_cls'])

    def build_predictions_layer(self):
        # Assign rnn outputs.
        if self.use_temporal_latent_space and self.use_variational_pi:
            self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, self.gmm_z, self.q_pi, self.p_pi, self.out_mu, self.out_sigma, self.out_rho, self.out_pen, self.out_eoc = self.outputs
        elif self.use_temporal_latent_space:
            self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, self.gmm_z, self.q_pi, self.out_mu, self.out_sigma, self.out_rho, self.out_pen, self.out_eoc = self.outputs
        elif self.use_variational_pi:
            self.gmm_z, self.q_pi, self.p_pi, self.out_mu, self.out_sigma, self.out_rho, self.out_pen, self.out_eoc = self.outputs

        # TODO: Sampling option.
        self.output_sample = tf.concat([self.out_mu, tf.round(self.out_pen)], axis=2)
        self.input_sample = self.inputs
        self.output_dim = self.output_sample.shape.as_list()[-1]

        # For analysis.
        self.norm_p_mu = tf.norm(self.p_mu, axis=-1)
        self.norm_p_sigma = tf.norm(self.p_sigma, axis=-1)
        self.norm_q_mu = tf.norm(self.q_mu, axis=-1)
        self.norm_q_sigma = tf.norm(self.q_sigma, axis=-1)
        self.norm_out_mu = tf.norm(self.out_mu, axis=-1)
        self.norm_out_sigma = tf.norm(self.out_sigma, axis=-1)

        self.ops_evaluation['output_sample'] = self.output_sample
        if self.use_temporal_latent_space:
            self.ops_evaluation['p_mu'] = self.p_mu
            self.ops_evaluation['p_sigma'] = self.p_sigma
            self.ops_evaluation['q_mu'] = self.q_mu
            self.ops_evaluation['q_sigma'] = self.q_sigma
        if self.use_variational_pi:
            self.ops_evaluation['p_pi'] = tf.nn.softmax(self.p_pi, axis=-1)
        self.ops_evaluation['q_pi'] = tf.nn.softmax(self.q_pi, axis=-1)

        self.ops_evaluation['gmm_z'] = self.gmm_z
        self.ops_evaluation['state'] = self.output_state
        self.ops_evaluation['out_eoc'] = self.out_eoc

        # In case we want to draw samples from output distribution instead of using mean.
        self.ops_evaluation['out_mu'] = self.out_mu
        self.ops_evaluation['out_sigma'] = self.out_sigma
        self.ops_evaluation['out_rho'] = self.out_rho
        self.ops_evaluation['out_pen'] = self.out_pen

        # Visualize average gmm sigma values.
        if self.is_gmm_active:
            self.ops_scalar_summary["mean_gmm_sigma"] = tf.reduce_mean(self.gmm_sigma)

        # Sequence mask for precise loss calculation.
        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_length, maxlen=tf.reduce_max(self.input_seq_length), dtype=tf.float32), -1)

    def build_loss(self):
        if self.is_training or self.is_validation:
            targets_mu = self.target_pieces[0]
            targets_pen = self.target_pieces[2]
            targets_eoc = self.target_pieces[3]

            if self.reconstruction_loss_key not in self.ops_loss:
                with tf.name_scope('reconstruction_loss'):
                    # Gaussian log likelihood loss (bivariate)
                    if self.reconstruction_loss == 'nll_normal_bi':
                        self.ops_loss[self.reconstruction_loss_key] = -self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_normal_bivariate(targets_mu, self.out_mu, self.out_sigma, self.out_rho, reduce_sum=False))
                    # Gaussian log likelihood loss (diagonal covariance)
                    elif self.reconstruction_loss == 'nll_normal_diag':
                        self.ops_loss[self.reconstruction_loss_key] = -self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_normal_diag_cov(targets_mu, self.out_mu, self.out_sigma, reduce_sum=False))
                    # L1 norm.
                    elif self.reconstruction_loss == "l1":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.absolute_difference(targets_mu, self.out_mu, reduction='none'))
                    # Mean-squared error.
                    elif self.reconstruction_loss == "mse":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.mean_squared_error(targets_mu, self.out_mu, reduction='none'))
                    else:
                        raise Exception("Undefined loss.")

            if "loss_pen" not in self.ops_loss:
                with tf.name_scope('pen_reconstruction_loss'):
                    # Bernoulli loss for pen information.
                    self.ops_loss['loss_pen'] = -self.pen_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_bernoulli(targets_pen, self.out_pen, reduce_sum=False))

            if "loss_eoc" not in self.ops_loss:
                with tf.name_scope('eoc_loss'):
                    self.ops_loss['loss_eoc'] = -self.eoc_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_bernoulli(targets_eoc, self.out_eoc, reduce_sum=False))

            VRNNGMM.build_loss(self)

    ########################################
    # Evaluation methods.
    ########################################
    def sample_func(self, session, seq_len, prev_state, ops_eval, text, eoc_threshold, cursive_threshold, use_sample_mean=True):
        """
        Sampling function generating new samples randomly by sampling one stroke at a time.

        Args:
            session:
            seq_len: # of frames.
            ops_eval: ops to be evaluated by the model.

        Returns:

        """
        cursive_style = False
        if cursive_threshold > 0.5:
            cursive_style = True

        if ops_eval is None:
            ops_eval = self.ops_evaluation
        # These ops are required by the sampling function.
        if 'out_eoc' not in ops_eval:
            ops_eval['out_eoc'] = self.out_eoc
        if 'output_sample' not in ops_eval:
            ops_eval['output_sample'] = self.output_sample
        if 'state' not in ops_eval:
            ops_eval['state'] = self.output_state

        # Since we draw one sample at a time, we need to accumulate the results.
        output_container = {}
        for key, val in ops_eval.items():
            output_container[key] = []

        def one_step(feed_dict, save=True):
            eval_results = session.run(ops_eval, feed_dict)

            if save or (eval_results['output_sample'][0, 0, 2] == 1):
                for key in output_container.keys():
                    output_container[key].append(eval_results[key])

                if use_sample_mean is False:
                    sigma1, sigma2 = np.square(eval_results['out_sigma'][0,0])
                    correlation = eval_results['out_rho'][0,0,0]*sigma1*sigma2
                    cov_matrix = [[sigma1, correlation], [correlation, sigma2]]
                    stroke_sample = np.reshape(np.random.multivariate_normal(eval_results['out_mu'][0][0], cov_matrix), (1,1,-1))
                    output_container['output_sample'][-1] = np.concatenate([stroke_sample, np.round(eval_results['out_pen'])], axis=-1)

            return eval_results['out_eoc'], eval_results['output_sample'], eval_results['state']

        use_bow_labels = self.use_bow_labels

        def prepare_model_input(char_label, bow_label):
            if use_bow_labels:
                return np.concatenate([np.zeros((1, 1, 3)), char_label, bow_label], axis=-1)
            else:
                return np.concatenate([np.zeros((1, 1, 3)), char_label], axis=-1)

        zero_char_label = np.zeros((1, 1, 70))
        bow_label = np.ones((1, 1, 1))
        non_bow_label = np.zeros((1, 1, 1))

        words = text.split(' ')

        prev_eoc_step = 0
        step = 0
        for word_idx, word in enumerate(words):
            char_idx = 0

            text_char_labels = np.reshape(self.text_to_label_fn(list(word)), (len(word), 1, 1, -1))
            char_label = zero_char_label

            prev_x = prepare_model_input(char_label, bow_label)

            last_step = False
            while step < seq_len:
                if last_step:
                    break
                step += 1
                feed = {self.inputs          : prev_x,
                        self.input_seq_length: np.ones(1),
                        self.initial_state   : prev_state}

                eoc, output_stroke, prev_state = one_step(feed_dict=feed)

                if np.squeeze(eoc) > eoc_threshold and (step - prev_eoc_step) > 4:
                    prev_eoc_step = step

                    char_idx += 1
                    if char_idx == len(word):
                        last_step = True
                        char_idx -= 1

                    if last_step or (not cursive_style):
                        # Peek one step ahead with blank step.
                        prev_x = prepare_model_input(zero_char_label, non_bow_label)

                        step += 1
                        feed = {self.inputs          : prev_x,
                                self.input_seq_length: np.ones(1),
                                self.initial_state   : prev_state}

                        eoc, output_stroke, prev_state = one_step(feed_dict=feed, save=last_step)

                prev_x = prepare_model_input(text_char_labels[char_idx], non_bow_label)
        # Concatenate output lists.
        for key, val in output_container.items():
            output_container[key] = np.concatenate(val, axis=1)

        return output_container

    def sample_biased(self, session, seq_len, prev_state, prev_sample=None, ops_eval=None, **kwargs):
        """
        Args:
            session:
            seq_len:
            prev_state:
            prev_sample:
            ops_eval:
            **kwargs:

        Returns:

        """

        text = kwargs.get('conditional_inputs', 'test, Function. Example')
        eoc_threshold = kwargs.get('eoc_threshold', 0.15)
        cursive_threshold = kwargs.get('cursive_threshold', 0.10)
        use_sample_mean = kwargs.get('use_sample_mean', True)

        ref_len = 0
        if prev_sample is not None:
            prev_sample = np.expand_dims(prev_sample, axis=0) if prev_sample.ndim == 2 else prev_sample
            ref_len = prev_sample.shape[1]
        seq_len = seq_len - ref_len

        output_container = self.sample_func(session, seq_len, prev_state, ops_eval, text, eoc_threshold, cursive_threshold, use_sample_mean)

        if prev_sample is not None:
            last_prev_sample_step = np.expand_dims(prev_sample[:, -1, :].copy(), axis=0)
            last_prev_sample_step[0,0,2] = 1.0
            output_container['output_sample'][0,0,0] = output_container['output_sample'][0,0,0] + 20
            output_container['output_sample'] = np.concatenate((prev_sample, last_prev_sample_step, output_container['output_sample']), axis=1)

        return [output_container]

    def sample_unbiased(self, session, seq_len=500, ops_eval=None, **kwargs):
        """
        Args:
            session:
            seq_len:
            ops_eval:
            **kwargs:

        Returns:
        """
        text = kwargs.get('conditional_inputs', 'test, Function. Example')
        eoc_threshold = kwargs.get('eoc_threshold', 0.15)
        cursive_threshold = kwargs.get('cursive_threshold', 0.10)
        use_sample_mean = kwargs.get('use_sample_mean', True)

        prev_state = session.run(self.cell.zero_state(batch_size=1, dtype=tf.float32))
        output_container = self.sample_func(session, seq_len, prev_state, ops_eval, text, eoc_threshold, cursive_threshold, use_sample_mean)

        return [output_container]
