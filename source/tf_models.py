import tensorflow as tf
import numpy as np
import sys
import time
import tf_loss
from tf_rnn_cells import VRNNCell, VRNNGmmCell
from tf_model_utils import get_reduce_loss_func

"""
Vanilla variational recurrent neural network model. Assuming that model outputs are isotropic Gaussian distributions.
The model is trained by using negative log-likelihood (reconstruction) and KL-divergence losses.

Model functionality is decomposed into basic functions (see build_graph method) so that variants of the model can easily
be implemented by inheriting from the vanilla architecture.
"""


class VRNN():
    def __init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=-1, mode="training"):

        self.config = config
        assert mode in ["training", "validation", "sampling"]
        self.mode = mode
        self.is_sampling = mode == "sampling"
        self.is_validation = mode == "validation"
        self.is_training = mode == "training"
        self.reuse = reuse

        self.inputs = input_op
        self.targets = target_op
        self.input_seq_length = input_seq_length_op
        self.input_dims = input_dims

        if target_op is not None or self.is_training or self.is_validation:
            self.target_dims = target_dims
            self.target_pieces = tf.split(self.targets, target_dims, axis=2)

        self.latent_size = self.config['latent_size']

        self.batch_size = config['batch_size'] if batch_size == -1 else batch_size

        # Reconstruction loss can be modeled differently. Create a key dynamically since the key is used in printing.
        self.reconstruction_loss = self.config.get('reconstruction_loss', 'nll_normal')
        self.reconstruction_loss_key = "loss_" + self.reconstruction_loss
        self.reconstruction_loss_weight = self.config.get('loss_weights', {}).get('reconstruction_loss', 1)
        self.kld_loss_weight = self.config.get('loss_weights', {}).get('kld_loss', 1)

        # Function to get final loss value: average loss or summation.
        self.reduce_loss_func = get_reduce_loss_func(self.config['reduce_loss'], self.input_seq_length)
        self.mean_sequence_func = get_reduce_loss_func("mean_per_step", self.input_seq_length)

        # TODO: Create a dictionary just for cell arguments.
        self.vrnn_cell_args = config
        self.vrnn_cell_args['input_dims'] = self.input_dims

        # To keep track of operations. List of graph nodes that must be evaluated by session.run during training.
        self.ops_loss = {}
        # Loss ops that are used to train the model.
        self.ops_training_loss = {}
        # (Default) graph ops to be fed into session.run while evaluating the model. Note that tf_evaluate* codes assume
        # to get at least these op results.
        self.ops_evaluation = {}
        # Graph ops for scalar summaries such as average predicted variance.
        self.ops_scalar_summary = {}

    def build_graph(self):
        self.get_constructors()
        self.build_cell()
        self.build_rnn_layer()
        self.build_predictions_layer()
        self.build_loss()
        self.accumulate_loss()
        self.create_summary_plots()
        self.log_num_parameters()

    def get_constructors(self):
        """
        Enables loading project specific classes.
        """
        self.vrnn_cell_constructor = getattr(sys.modules[__name__], self.config['vrnn_cell_cls'])

    def build_cell(self):
        if self.mode == "training" or self.mode == "validation":
            self.cell = self.vrnn_cell_constructor(reuse=self.reuse, mode=self.mode, config=self.vrnn_cell_args)
        elif self.mode == "sampling":
            self.cell = self.vrnn_cell_constructor(reuse=self.reuse, mode=self.mode, config=self.vrnn_cell_args)

        assert isinstance(self.cell, VRNNCell), "Cell object must be an instance of VRNNCell for VRNN model."

        self.initial_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_rnn_layer(self):
        # Get VRNN cell output
        if self.config['use_dynamic_rnn']:
            self.outputs, self.output_state = tf.nn.dynamic_rnn(self.cell,
                                                                self.inputs,
                                                                sequence_length=self.input_seq_length,
                                                                initial_state=self.initial_state,
                                                                dtype=tf.float32)
        else:
            inputs_static_rnn = tf.unstack(self.inputs, axis=1)
            self.outputs_static_rnn, self.output_state = tf.nn.static_rnn(self.cell,
                                                                     inputs_static_rnn,
                                                                     initial_state=self.initial_state,
                                                                     sequence_length=self.input_seq_length,
                                                                     dtype=tf.float32)

            self.outputs = [] # Parse static rnn outputs and convert them into the same format with dynamic rnn.
            if self.config['use_dynamic_rnn'] is False:
                for n, name in enumerate(self.config['output']['keys']):
                    x = tf.stack([o[n] for o in self.outputs_static_rnn], axis=1)
                    self.outputs.append(x)

    def build_predictions_layer(self):
        # Assign rnn outputs.
        self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, self.out_mu, self.out_sigma = self.outputs

        # TODO: Sampling option.
        self.output_sample = self.out_mu
        self.input_sample = self.inputs
        self.output_dim = self.output_sample.shape.as_list()[-1]

        self.ops_evaluation['output_sample'] = self.output_sample
        self.ops_evaluation['p_mu'] = self.p_mu
        self.ops_evaluation['p_sigma'] = self.p_sigma
        self.ops_evaluation['q_mu'] = self.q_mu
        self.ops_evaluation['q_sigma'] = self.q_sigma
        self.ops_evaluation['state'] = self.output_state

        num_entries = tf.cast(self.input_seq_length.shape.as_list()[0]*tf.reduce_sum(self.input_seq_length), tf.float32)
        self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.out_sigma) / num_entries
        self.ops_scalar_summary["mean_p_sigma"] = tf.reduce_sum(self.p_sigma) / num_entries
        self.ops_scalar_summary["mean_q_sigma"] = tf.reduce_sum(self.q_sigma) / num_entries

        # Mask for precise loss calculation.
        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_length, maxlen=tf.reduce_max(self.input_seq_length), dtype=tf.float32), -1)

    def build_loss(self):
        if self.is_training or self.is_validation:
            # TODO: Use dataset object to parse the concatenated targets.
            targets_mu = self.target_pieces[0]

            if self.reconstruction_loss_key not in self.ops_loss:
                with tf.name_scope('reconstruction_loss'):
                    # Gaussian log likelihood loss.
                    if self.reconstruction_loss == 'nll_normal_iso':
                        self.ops_loss[self.reconstruction_loss_key] = -self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_normal_diag_cov(targets_mu, self.out_mu, self.out_sigma, reduce_sum=False))
                    # L1 norm.
                    elif self.reconstruction_loss == "l1":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.absolute_difference(targets_mu, self.out_mu, reduction='none'))
                    # Mean-squared error.
                    elif self.reconstruction_loss == "mse":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.mean_squared_error(targets_mu, self.out_mu, reduction='none'))
                    else:
                        raise Exception("Undefined loss.")

            if "loss_kld" not in self.ops_loss:
                with tf.name_scope('kld_loss'):
                    self.ops_loss['loss_kld'] = self.kld_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.kld_normal_isotropic(self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, reduce_sum=False))

    def accumulate_loss(self):
        # Accumulate losses to create training optimization.
        # Model.loss is used by the optimization function.
        self.loss = 0
        for _, loss_op in self.ops_loss.items():
            self.loss += loss_op
        self.ops_loss['total_loss'] = self.loss

    def log_loss(self, eval_loss, step=0, epoch=0, time_elapsed=None, prefix=""):
        loss_format = prefix + "{}/{} \t Total: {:.4f} \t"
        loss_entries = [step, epoch, eval_loss['total_loss']]

        for loss_key in sorted(eval_loss.keys()):
            if loss_key != 'total_loss':
                loss_format += "{}: {:.4f} \t"
                loss_entries.append(loss_key)
                loss_entries.append(eval_loss[loss_key])

        if time_elapsed is not None:
            print(loss_format.format(*loss_entries) + "time/batch = {:.3f}".format(time_elapsed))
        else:
            print(loss_format.format(*loss_entries))

    def log_num_parameters(self):
        num_param = 0
        for v in tf.global_variables():
            num_param += np.prod(v.get_shape().as_list())

        self.num_parameters = num_param
        print("# of parameters: " + str(num_param))

    def create_summary_plots(self):
        """
        Creates scalar summaries for loss plots. Iterates through `ops_loss` member and create a summary entry.

        If the model is in `validation` mode, then we follow a different strategy. In order to have a consistent
        validation report over iterations, we first collect model performance on every validation mini-batch and then
        report the average loss. Due to tensorflow's lack of loss averaging ops, we need to create placeholders per loss
        to pass the average loss.

        Returns:

        """
        if self.is_training:
            for loss_name, loss_op in self.ops_loss.items():
                tf.summary.scalar(loss_name, loss_op, collections=[self.mode+'_summary_plot', self.mode+'_loss'])

        elif self.is_validation: # Validation: first accumulate losses and then log them.
            self.container_loss = {}
            self.container_loss_placeholders = {}
            self.container_validation_feed_dict = {}
            self.validation_summary_num_runs = 0

            for loss_name, _ in self.ops_loss.items():
                self.container_loss[loss_name] = 0
                self.container_loss_placeholders[loss_name] = tf.placeholder(tf.float32, shape=[])
                tf.summary.scalar(loss_name, self.container_loss_placeholders[loss_name], collections=[self.mode+'_summary_plot', self.mode+'_loss'])
                self.container_validation_feed_dict[self.container_loss_placeholders[loss_name]] = 0

        for summary_name, scalar_summary_op in self.ops_scalar_summary.items():
            tf.summary.scalar(summary_name, scalar_summary_op, collections=[self.mode+'_summary_plot', self.mode+'_scalar_summary'])

        # Create summaries to visualize distribution of latent variables.
        if self.config['tensorboard_verbose'] > 0:
            tf.summary.histogram("p_mu", self.p_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            tf.summary.histogram("p_sigma", self.p_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            tf.summary.histogram("q_mu", self.q_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            tf.summary.histogram("q_sigma", self.q_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            tf.summary.histogram("out_mu", self.out_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            tf.summary.histogram("out_sigma", self.out_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])

        self.loss_summary = tf.summary.merge_all(self.mode+'_summary_plot')

    ########################################
    # Summary methods for validation mode.
    ########################################
    def update_validation_loss(self, loss_evaluated):
        self.validation_summary_num_runs += 1
        for loss_name, loss_value in loss_evaluated.items():
            self.container_loss[loss_name] += loss_value

    def reset_validation_loss(self):
        for loss_name, loss_value in self.container_loss.items():
            self.container_loss[loss_name] = 0

    def get_validation_summary(self, session):
        for loss_name, loss_pl in self.container_loss_placeholders.items():
            self.container_loss[loss_name] /= self.validation_summary_num_runs
            self.container_validation_feed_dict[loss_pl] = self.container_loss[loss_name]
        self.validation_summary_num_runs = 0

        # return self.container_validation_feed_dict, self.container_loss
        valid_summary = session.run(self.loss_summary, self.container_validation_feed_dict)
        return valid_summary, self.container_loss

    ########################################
    # Evaluation methods.
    ########################################

    def reconstruct_given_sample(self, session, inputs, targets=None, ops_eval=None):
        """
        Reconstructs a given sample.

        Args:
            session:
            inputs: input tensor of size (batch_size, sequence_length, input_size).
            targets: to calculate model loss. if None, then loss is not calculated.
            ops_eval: ops to be evaluated by the model.

        Returns:

        """
        model_inputs = np.expand_dims(inputs, axis=0) if inputs.ndim == 2 else inputs
        model_targets = np.expand_dims(targets, axis=0) if (targets is not None) and (targets.ndim == 2) else targets
        eval_op_list = []
        if ops_eval is None:
            ops_eval = self.ops_evaluation
        eval_op_list.append(ops_eval)

        feed = {self.inputs          : model_inputs,
                self.input_seq_length: np.ones(1)*model_inputs.shape[1]}

        if model_targets is not None:
            feed[self.targets] = model_targets
            eval_op_list.append(self.ops_loss)

        eval_results = session.run(eval_op_list, feed)

        return eval_results

    def sample_unbiased(self, session, seq_len=500, ops_eval=None, **kwargs):
        """
        Generates new samples randomly. Note that this function works only if the model is created in "sampling" mode.

        Args:
            **kwargs:
            session:
            seq_len: # of frames.
            ops_eval: ops to be evaluated by the model.

        Returns:

        """
        dummy_x = np.zeros((self.batch_size, seq_len, sum(self.input_dims)))
        prev_state = session.run(self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32))

        eval_op_list = []
        if ops_eval is None:
            ops_eval = self.ops_evaluation
        eval_op_list.append(ops_eval)

        model_inputs = dummy_x
        feed = {self.inputs          : model_inputs,
                self.input_seq_length: np.ones(1)*model_inputs.shape[1],
                self.initial_state: prev_state}

        eval_results = session.run(eval_op_list, feed)
        return eval_results

    def sample_biased(self, session, seq_len, prev_state, prev_sample=None, ops_eval=None, **kwargs):
        """
        Initializes the model by using state of a real sample.

        Args:
            session:
            seq_len:
            prev_state: rnn state to be used as reference.
            prev_sample: sample that is used to bias the model and generate prev_state. If not None, then it is
                concatenated with the synthetic sample for visualization.
            ops_eval: ops to be evaluated by the model.

        Returns:
        """

        ref_len = 0
        if prev_sample is not None:
            prev_sample = np.expand_dims(prev_sample, axis=0) if prev_sample.ndim == 2 else prev_sample
            ref_len = prev_sample.shape[1]

            output_sample_concatenated = np.zeros((self.batch_size, seq_len, self.output_dim), dtype=np.float32)
            output_sample_concatenated[:, :ref_len] = prev_sample[:, :ref_len]  # ref_sample_reconstructed

        seq_len = seq_len - ref_len
        dummy_x = np.zeros((self.batch_size, seq_len, sum(self.input_dims)))

        eval_op_list = []
        if ops_eval is None:
            ops_eval = self.ops_evaluation
        eval_op_list.append(ops_eval)

        model_inputs = dummy_x
        feed = {self.inputs          : model_inputs,
                self.input_seq_length: np.ones(1)*model_inputs.shape[1],
                self.initial_state   : prev_state}

        eval_results = session.run(eval_op_list, feed)

        if prev_sample is not None:
            output_sample_concatenated[:, ref_len:] = eval_results[0]['output_sample']
            eval_results[0]['output_sample'] = output_sample_concatenated

        return eval_results


class VRNNGMM(VRNN):
    def __init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=-1, mode="training"):
        VRNN.__init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=batch_size, mode=mode)

        # VRNNCellGMM configuration.
        self.use_temporal_latent_space = config.get('use_temporal_latent_space', True)
        self.use_variational_pi = config.get('use_variational_pi', False)
        self.use_real_pi_labels = config.get('use_real_pi_labels', False)
        self.use_soft_gmm = config.get('use_soft_gmm', False)
        self.is_gmm_active = not(config.get('use_pi_as_content', False))

        self.num_gmm_components = config['num_gmm_components']
        self.gmm_component_size = config['gmm_component_size']

        self.kld_loss_pi_weight = self.config.get('loss_weights', {}).get('kld_loss_pi', 1)
        self.gmm_sigma_regularizer_weight = self.config.get('loss_weights', {}).get('gmm_sigma_regularizer', None)
        self.classification_loss_weight = self.config.get('loss_weights', {}).get('classification_loss', None)
        self.pi_entropy_loss_weight = self.config.get('loss_weights', {}).get('pi_entropy_loss', None)

        self.use_classification_loss = False if self.classification_loss_weight is None else True
        self.use_gmm_sigma_loss = False if self.gmm_sigma_regularizer_weight is None else True
        self.use_pi_entropy_loss = False if self.pi_entropy_loss_weight is None else True

        # Sanity Check
        if target_op is not None or self.is_training or self.is_validation:
            assert not (self.use_real_pi_labels and len(self.target_dims) < 2), "Real labels are not provided: rank(target_dims) < 2."
            assert not (self.use_classification_loss and len(self.target_dims) < 2), "Real labels are not provided for classification loss: rank(target_dims) < 2."

    def build_cell(self):
        if self.mode == "training" or self.mode == "validation":
            self.cell = self.vrnn_cell_constructor(reuse=self.reuse, mode=self.mode, config=self.vrnn_cell_args)
        elif self.mode == "sampling":
            self.cell = self.vrnn_cell_constructor(reuse=self.reuse, mode=self.mode, config=self.vrnn_cell_args)

        assert isinstance(self.cell, VRNNGmmCell), "Cell object must be an instance of VRNNCellGMM for VRNNGMM model."

        # GMM components are 2D: [# components, component size]
        if self.is_gmm_active:
            self.gmm_mu, self.gmm_sigma = self.cell.get_gmm_components()

        self.initial_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_predictions_layer(self):
        # Assign rnn outputs.
        if self.use_temporal_latent_space and self.use_variational_pi:
            self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, self.gmm_z, self.q_pi, self.p_pi, self.out_mu, self.out_sigma = self.outputs
        elif self.use_temporal_latent_space:
            self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, self.gmm_z, self.q_pi, self.out_mu, self.out_sigma = self.outputs
        elif self.use_variational_pi:
            self.gmm_z, self.q_pi, self.p_pi, self.out_mu, self.out_sigma = self.outputs

        # TODO: Sampling option.
        self.output_sample = self.out_mu
        self.input_sample = self.inputs
        self.output_dim = self.output_sample.shape.as_list()[-1]

        self.ops_evaluation['output_sample'] = self.output_sample
        if self.use_temporal_latent_space:
            self.ops_evaluation['p_mu'] = self.p_mu
            self.ops_evaluation['p_sigma'] = self.p_sigma
            self.ops_evaluation['q_mu'] = self.q_mu
            self.ops_evaluation['q_sigma'] = self.q_sigma
        if self.use_variational_pi:
            self.ops_evaluation['p_pi'] = tf.nn.softmax(self.p_pi, dim=-1)
        self.ops_evaluation['q_pi'] = tf.nn.softmax(self.q_pi, dim=-1)
        self.ops_evaluation['gmm_z'] = self.gmm_z
        self.ops_evaluation['state'] = self.output_state

        num_entries = tf.cast(self.input_seq_length.shape.as_list()[0]*tf.reduce_sum(self.input_seq_length), tf.float32)
        self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.out_sigma)/num_entries
        self.ops_scalar_summary["mean_p_sigma"] = tf.reduce_sum(self.p_sigma)/num_entries
        self.ops_scalar_summary["mean_q_sigma"] = tf.reduce_sum(self.q_sigma)/num_entries

        # Mask for precise loss calculation.
        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_length, maxlen=tf.reduce_max(self.input_seq_length), dtype=tf.float32), -1)

    def build_loss(self):
        if self.is_training or self.is_validation:
            # TODO: Use dataset object to parse the concatenated targets.
            targets_mu = self.target_pieces[0]

            if self.reconstruction_loss_key not in self.ops_loss:
                with tf.name_scope('reconstruction_loss'):
                    # Gaussian log likelihood loss.
                    if self.reconstruction_loss == 'nll_normal_iso':
                        self.ops_loss[self.reconstruction_loss_key] = -self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.logli_normal_diag_cov(targets_mu, self.out_mu, self.out_sigma, reduce_sum=False))
                    # L1 norm.
                    elif self.reconstruction_loss == "l1":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.absolute_difference(targets_mu, self.out_mu, reduction='none'))
                    # Mean-squared error.
                    elif self.reconstruction_loss == "mse":
                        self.ops_loss[self.reconstruction_loss_key] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.losses.mean_squared_error(targets_mu, self.out_mu, reduction='none'))
                    else:
                        raise Exception("Undefined loss.")

            if self.use_temporal_latent_space and not "loss_kld" in self.ops_loss:
                with tf.name_scope('kld_loss'):
                    self.ops_loss['loss_kld'] = self.kld_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf_loss.kld_normal_isotropic(self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, reduce_sum=False))

            flat_q_pi = tf.reshape(self.q_pi, [-1, self.num_gmm_components])
            self.dist_q = tf.contrib.distributions.Categorical(logits=flat_q_pi)

            if self.use_variational_pi and not "loss_kld_pi" in self.ops_loss:
                with tf.name_scope('kld_pi_loss'):
                    flat_p_pi = tf.reshape(self.p_pi, [-1, self.num_gmm_components])
                    self.dist_p = tf.contrib.distributions.Categorical(logits=flat_p_pi)

                    flat_kld_cat_loss = tf.contrib.distributions.kl_divergence(distribution_a=self.dist_q, distribution_b=self.dist_p)
                    temporal_kld_cat_loss = tf.reshape(flat_kld_cat_loss, [self.batch_size, -1, 1])
                    self.ops_loss["loss_kld_pi"] = self.kld_loss_pi_weight*self.reduce_loss_func(self.seq_loss_mask*temporal_kld_cat_loss)

            if self.use_pi_entropy_loss and not "loss_entropy_pi" in self.ops_loss:
                self.ops_loss["loss_entropy_pi"] = self.reconstruction_loss_weight*self.reduce_loss_func(self.seq_loss_mask*tf.reshape(self.dist_q.entropy(), [self.batch_size, -1, 1]))


            if self.use_classification_loss and not "loss_classification" in self.ops_loss:
                targets_categorical_labels = self.target_pieces[1]
                # Use GMM latent space probabilities as class predictions.
                self.label_predictions = self.q_pi

                with tf.name_scope('classification_loss'):
                    prediction_size = targets_categorical_labels.get_shape().as_list()[-1]
                    flat_labels = tf.reshape(targets_categorical_labels, [-1, prediction_size])
                    flat_predictions = tf.reshape(self.label_predictions, [-1, prediction_size])

                    flat_char_classification_loss = tf.losses.softmax_cross_entropy(flat_labels, flat_predictions, reduction="none")
                    temporal_char_classification_loss = tf.reshape(flat_char_classification_loss, [self.batch_size, -1, 1])
                    self.ops_loss["loss_classification"] = self.classification_loss_weight*self.reduce_loss_func(self.seq_loss_mask*temporal_char_classification_loss)

            if self.is_gmm_active and self.use_gmm_sigma_loss and not "loss_gmm_sigma" in self.ops_loss:
                with tf.name_scope('gmm_sigma_loss'):
                    self.ops_loss["loss_gmm_sigma"] = tf.reduce_mean(tf.square(1 - self.gmm_sigma))

    def create_summary_plots(self):
        """
        Creates scalar summaries for loss plots. Iterates through `ops_loss` member and create a summary entry.

        If the model is in `validation` mode, then we follow a different strategy. In order to have a consistent
        validation report over iterations, we first collect model performance on every validation mini-batch and then
        report the average loss. Due to tensorflow's lack of loss averaging ops, we need to create placeholders per loss
        to pass the average loss.

        Returns:

        """
        if self.is_training:
            for loss_name, loss_op in self.ops_loss.items():
                tf.summary.scalar(loss_name, loss_op, collections=[self.mode+'_summary_plot', self.mode+'_loss'])

        elif self.is_validation: # Validation: first accumulate losses and then log them.
            self.container_loss = {}
            self.container_loss_placeholders = {}
            self.container_validation_feed_dict = {}
            self.validation_summary_num_runs = 0

            for loss_name, _ in self.ops_loss.items():
                self.container_loss[loss_name] = 0
                self.container_loss_placeholders[loss_name] = tf.placeholder(tf.float32, shape=[])
                tf.summary.scalar(loss_name, self.container_loss_placeholders[loss_name], collections=[self.mode+'_summary_plot', self.mode+'_loss'])
                self.container_validation_feed_dict[self.container_loss_placeholders[loss_name]] = 0

        for summary_name, scalar_summary_op in self.ops_scalar_summary.items():
            tf.summary.scalar(summary_name, scalar_summary_op, collections=[self.mode + '_summary_plot', self.mode + '_scalar_summary'])

        # Create summaries to visualize distribution of latent variables.
        if self.config['tensorboard_verbose'] > 0:
            if self.is_gmm_active:
                tf.summary.histogram("gmm_mu", self.gmm_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
                tf.summary.histogram("gmm_sigma", self.gmm_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            if self.use_temporal_latent_space:
                tf.summary.histogram("p_mu", self.p_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
                tf.summary.histogram("p_sigma", self.p_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
                tf.summary.histogram("q_mu", self.q_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
                tf.summary.histogram("q_sigma", self.q_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            if self.use_variational_pi:
                tf.summary.histogram("p_pi", tf.nn.softmax(self.p_pi), collections=[self.mode + '_summary_plot', self.mode + '_stochastic_variables'])

            tf.summary.histogram("q_pi", tf.nn.softmax(self.q_pi), collections=[self.mode + '_summary_plot', self.mode + '_stochastic_variables'])
            tf.summary.histogram("out_mu", self.out_mu, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])
            tf.summary.histogram("out_sigma", self.out_sigma, collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])

        self.loss_summary = tf.summary.merge_all(self.mode+'_summary_plot')

    def evaluate_gmm_latent_space(self, session):
        gmm_mus, gmm_sigmas = session.run([self.gmm_mu, self.gmm_sigma])
        return gmm_mus, gmm_sigmas


