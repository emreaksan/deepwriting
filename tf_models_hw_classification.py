import tensorflow as tf
import numpy as np

import tf_loss
from tf_model_utils import fully_connected_layer, linear, get_reduce_loss_func, get_rnn_cell

"""
Handwriting Classification/Segmentation Models
"""

class RNNClassifier():
    """
    Recurrent neural network with additional input and output fully connected layers.
    """
    def __init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=-1, mode="training"):

        self.config = config
        assert mode in ["training", "validation"]
        self.mode = mode
        self.is_training = mode == "training"
        self.is_validation = mode == "validation"
        self.reuse = reuse

        self.inputs = input_op
        self.targets = target_op
        self.input_seq_length = input_seq_length_op
        self.input_dims = input_dims
        self.target_dims = target_dims
        self.target_pieces = tf.split(self.targets, target_dims, axis=2)

        self.batch_size = config['batch_size'] if batch_size < 1 else batch_size

        # Function to get final loss value: average loss or summation.
        self.reduce_loss_func = get_reduce_loss_func(self.config['reduce_loss'], self.input_seq_length)
        self.mean_sequence_func = get_reduce_loss_func("mean_per_step", self.input_seq_length)

        self.weight_classification_loss = self.config.get('loss_weights', {}).get('classification_loss', 1)
        self.weight_eoc_loss = self.config.get('loss_weights', {}).get('eoc_loss', 1)
        self.weight_bow_loss = self.config.get('loss_weights', {}).get('bow_loss', 1)

        self.input_layer_config = config['input_layer']
        self.rnn_config = config['rnn_layer']
        self.output_layer_config = config['output_layer']

        if self.output_layer_config['out_dims'] is None:
            self.output_layer_config['out_dims'] = self.target_dims
        else:
            assert self.output_layer_config['out_dims'] == self.target_dims, "Output layer dimensions don't match with dataset target dimensions."

        # To keep track of operations. List of graph nodes that must be evaluated by session.run during training.
        self.ops_loss = {}
        # (Default) graph ops to be fed into session.run while evaluating the model. Note that tf_evaluate* codes expect
        # to get these op results. `log_loss` method also uses the same evaluated results.
        self.ops_evaluation = {}
        # Graph ops for scalar summaries such as accuracy.
        self.ops_scalar_summary = {}


    def flat_tensor(self, tensor, dim=-1):
        """
        Reshapes a tensor such that it has 2 dimensions. The dimension specified by `dim` is kept.
        """
        keep_dim_size = tensor.get_shape().as_list()[dim]
        return tf.reshape(tensor, [-1, keep_dim_size])


    def temporal_tensor(self, flat_tensor):
        """
        Reshapes a flat tensor (2-dimensional) to a tensor with shape (batch_size, seq_len, feature_size). Assuming
        that the flat tensor is with shape (batch_size*seq_len, feature_size)
        """
        feature_size = flat_tensor.get_shape().as_list()[1]
        return tf.reshape(flat_tensor, [self.batch_size, -1, feature_size])


    def build_graph(self):
        """
        Builds model and creates plots for tensorboard. Decomposes model building into sub-modules and makes inheritance
        is easier.
        """
        self.create_cells()
        self.build_input_layer()
        self.build_rnn_layer()
        self.build_output_layer()
        self.build_loss()
        self.accumulate_loss()
        self.create_scalar_summary()
        self.log_num_parameters()


    def create_cells(self):
        """
        Creates a Tensorflow RNN cell object by using the given configuration.
        """
        self.cell = get_rnn_cell(scope='rnn_cell', reuse=self.reuse, **self.rnn_config)
        self.initial_states = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)


    def build_input_layer(self):
        """
        Builds a number fully connected layers projecting the inputs into an embedding space. It was reported to be
        useful.
        """
        if self.input_layer_config is not None:
            with tf.variable_scope('input_layer', reuse=self.reuse):
                flat_inputs_hidden = self.flat_tensor(self.inputs)
                flat_inputs_hidden = fully_connected_layer(flat_inputs_hidden, **self.input_layer_config)

            self.inputs_hidden = self.temporal_tensor(flat_inputs_hidden)


    def build_rnn_layer(self):
        """
        Builds RNN layer by using dynamic_rnn wrapper of Tensorflow.
        """
        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.outputs, self.output_state = tf.nn.dynamic_rnn(self.cell,
                                                                self.inputs_hidden,
                                                                sequence_length=self.input_seq_length,
                                                                initial_state=self.initial_states,
                                                                dtype=tf.float32)

    def build_output_layer(self):
        """
        Builds a number fully connected layers projecting RNN predictions into an embedding space. Then, for each model
        output is predicted by a linear layer.
        """
        flat_outputs_hidden = self.flat_tensor(self.outputs)
        with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
            flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, **self.output_layer_config)

        with tf.variable_scope("output_layer_char", reuse=self.reuse):
            self.flat_char_prediction = linear(input=flat_outputs_hidden,
                                               output_size=self.target_dims[0],
                                               activation_fn=self.output_layer_config['out_activation_fn'][0],
                                               is_training=self.is_training)
            self.char_prediction = self.temporal_tensor(self.flat_char_prediction)

        with tf.variable_scope("output_layer_eoc", reuse=self.reuse):
            self.flat_eoc_prediction = linear(input=flat_outputs_hidden,
                                               output_size=self.target_dims[1],
                                               activation_fn=self.output_layer_config['out_activation_fn'][1],
                                               is_training=self.is_training)
            self.eoc_prediction = self.temporal_tensor(self.flat_eoc_prediction)

        with tf.variable_scope("output_layer_bow", reuse=self.reuse):
            self.flat_bow_prediction = linear(input=flat_outputs_hidden,
                                               output_size=self.target_dims[2],
                                               activation_fn=self.output_layer_config['out_activation_fn'][2],
                                               is_training=self.is_training)
            self.bow_prediction = self.temporal_tensor(self.flat_bow_prediction)

        # Mask for precise loss calculation.
        self.input_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_length, maxlen=tf.reduce_max(self.input_seq_length), dtype=tf.float32), -1)

        self.ops_evaluation['char_prediction'] = self.char_prediction
        self.ops_evaluation['eoc_prediction'] = self.eoc_prediction
        self.ops_evaluation['bow_prediction'] = self.bow_prediction


    def build_loss(self):
        """
        Builds loss terms:
        (1) cross-entropy loss for character classification,
        (2) bernoulli loss for end-of-character label prediction,
        (3) bernoulli loss for beginning-of-word label prediction.
        """
        with tf.name_scope('cross_entropy_char_loss'):
            flat_char_targets = tf.reshape(self.target_pieces[0], [-1, self.target_dims[0]])
            flat_char_classification_loss = tf.losses.softmax_cross_entropy(flat_char_targets, self.flat_char_prediction, reduction="none")
            char_classification_loss = tf.reshape(flat_char_classification_loss, [self.batch_size, -1, 1])
            self.ops_loss['loss_cross_entropy_char'] = self.weight_classification_loss*self.reduce_loss_func(self.input_mask*char_classification_loss)

        with tf.name_scope('bernoulli_eoc_loss'):
            self.ops_loss['loss_bernoulli_eoc'] = -self.weight_eoc_loss*self.reduce_loss_func(self.input_mask*tf_loss.logli_bernoulli(self.target_pieces[1], self.eoc_prediction, reduce_sum=False))

        with tf.name_scope('bernoulli_bow_loss'):
            self.ops_loss['loss_bernoulli_bow'] = -self.weight_bow_loss*self.reduce_loss_func(self.input_mask*tf_loss.logli_bernoulli(self.target_pieces[2], self.bow_prediction, reduce_sum=False))

        # Accuracy
        predictions = tf.argmax(self.flat_char_prediction, 1)
        targets = tf.argmax(flat_char_targets, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), tf.float32))
        self.ops_scalar_summary['accuracy'] = self.accuracy


    def accumulate_loss(self):
        """
        Accumulate losses to create training optimization. Model.loss is used by the optimization function.
        """
        self.loss = 0
        for _, loss_op in self.ops_loss.items():
            self.loss += loss_op
        self.ops_loss['total_loss'] = self.loss


    def log_loss(self, eval_loss, step=0, epoch=0, time_elapsed=None, prefix=""):
        """
        Prints status messages during training. It is called in the main training loop.
        Args:
            eval_loss (dict): evaluated results of `ops_loss` dictionary.
            step (int): current step.
            epoch (int): current epoch.
            time_elapsed (float): elapsed time.
            prefix (str): some informative text. For example, "training" or "validation".
        """
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
        """
        Prints total number of parameters.
        """
        num_param = 0
        for v in tf.global_variables():
            num_param += np.prod(v.get_shape().as_list())

        self.num_parameters = num_param
        print("# of parameters: " + str(num_param))


    def create_scalar_summary(self):
        """
        Creates scalar summaries for loss plots. Iterates through `ops_loss` member and create a summary entry.

        If the model is in `validation` mode, then we follow a different strategy. In order to have a consistent
        validation report over iterations, we first collect model performance on every validation mini-batch
        and then report the average loss. Due to tensorflow's lack of loss averaging ops, we need to create
        placeholders per loss to pass the average loss.
        """
        if self.is_training:
            # For each loss term, create a tensorboard plot.
            for loss_name, loss_op in self.ops_loss.items():
                tf.summary.scalar(loss_name, loss_op, collections=[self.mode + '_summary_plot', self.mode + '_loss'])

        else:
            # Validation: first accumulate losses and then plot.
            # Create containers and placeholders for every loss term. After each validation step, keeps summing losses.
            # At the end of validation loop, calculates average performance on the whole validation dataset and creates
            # summary entries.
            self.container_loss = {}
            self.container_loss_placeholders = {}
            self.container_loss_summaries = {}
            self.container_validation_feed_dict = {}
            self.validation_summary_num_runs = 0

            for loss_name, _ in self.ops_loss.items():
                self.container_loss[loss_name] = 0
                self.container_loss_placeholders[loss_name] = tf.placeholder(tf.float32, shape=[])
                tf.summary.scalar(loss_name, self.container_loss_placeholders[loss_name], collections=[self.mode + '_summary_plot', self.mode + '_loss'])
                self.container_validation_feed_dict[self.container_loss_placeholders[loss_name]] = 0.0

        for summary_name, scalar_summary_op in self.ops_scalar_summary.items():
            tf.summary.scalar(summary_name, scalar_summary_op, collections=[self.mode + '_summary_plot', self.mode + '_scalar_summary'])

        self.loss_summary = tf.summary.merge_all(self.mode + '_summary_plot')

    ########################################
    # Summary methods for validation mode.
    ########################################
    def update_validation_loss(self, loss_evaluated):
        """
        Updates validation losses. Note that this method is called after every validation step.

        Args:
            loss_evaluated: valuated results of `ops_loss` dictionary.
        """
        self.validation_summary_num_runs += 1
        for loss_name, loss_value in loss_evaluated.items():
            self.container_loss[loss_name] += loss_value

    def reset_validation_loss(self):
        """
        Resets validation loss containers.
        """
        for loss_name, loss_value in self.container_loss.items():
            self.container_loss[loss_name] = 0

    def get_validation_summary(self):
        """
        Creates a feed dictionary of validation losses for validation summary. Note that this method is called after
        validation loops is over.

        Returns (dict, dict):
            feed_dict for validation summary.
            average `ops_loss` results for `log_loss` method.
        """
        for loss_name, loss_pl in self.container_loss_placeholders.items():
            self.container_loss[loss_name] /= self.validation_summary_num_runs
            self.container_validation_feed_dict[loss_pl] = self.container_loss[loss_name]

        self.validation_summary_num_runs = 0

        return self.container_validation_feed_dict, self.container_loss

    ########################################
    # Evaluation methods.
    ########################################
    def classify_given_sample(self, session, inputs, targets=None, ops_eval=None):
        """
        Classifies a given handwriting sample.

        Args:
            session:
            inputs: input tensor of size (batch_size, sequence_length, input_size).
            targets: to calculate model loss. if None, then loss is not calculated.
            ops_eval: ops to be evaluated by the model.

        Returns (list): the first element is a dictionary of evaluated graph ops and the second elements is a dictionary
            of losses if the `targets` is passed.
        """
        model_inputs = np.expand_dims(inputs, axis=0) if inputs.ndim == 2 else inputs
        model_targets = np.expand_dims(targets, axis=0) if (targets is not None) and (targets.ndim == 2) else targets
        eval_op_list = []
        if ops_eval is None:
            ops_eval = self.ops_evaluation
        eval_op_list.append(ops_eval)

        feed = {self.inputs    : model_inputs,
                self.input_seq_length: np.ones(1)*model_inputs.shape[1]}

        if model_targets is not None:
            feed[self.targets] = model_targets
            eval_op_list.append(self.ops_loss)

        eval_results = session.run(eval_op_list, feed)

        return eval_results


class BiDirectionalRNNClassifier(RNNClassifier):
    """
    Bidirectional recurrent neural network with additional input and output fully connected layers.
    """
    def __init__(self, config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size=-1, mode="training"):
        super(BiDirectionalRNNClassifier, self).__init__(config, input_op, input_seq_length_op, target_op, input_dims, target_dims, reuse, batch_size, mode)

        self.cells_fw = []
        self.cells_bw = []

        self.initial_states_fw = []
        self.initial_states_bw = []

        # See https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
        self.stack_fw_bw_cells = self.rnn_config.get('stack_fw_bw_cells', True)

    def create_cells(self):
        if self.stack_fw_bw_cells:
            single_cell_config = self.rnn_config.copy()
            single_cell_config['num_layers'] = 1
            for i in range(self.rnn_config['num_layers']):
                cell_fw = get_rnn_cell(scope='rnn_cell_fw', reuse=self.reuse, **single_cell_config)
                self.cells_fw.append(cell_fw)
                self.initial_states_fw.append(cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

                cell_bw = get_rnn_cell(scope='rnn_cell_bw', reuse=self.reuse, **single_cell_config)
                self.cells_bw.append(cell_bw)
                self.initial_states_bw.append(cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32))
        else:
            cell_fw = get_rnn_cell(scope='rnn_cell_fw', reuse=self.reuse, **self.rnn_config)
            self.cells_fw.append(cell_fw)
            self.initial_states_fw.append(cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

            cell_bw = get_rnn_cell(scope='rnn_cell_bw', reuse=self.reuse, **self.rnn_config)
            self.cells_bw.append(cell_bw)
            self.initial_states_bw.append(cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

    def build_rnn_layer(self):
        with tf.variable_scope("bidirectional_rnn_layer", reuse=self.reuse):
            if self.stack_fw_bw_cells:
                self.outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=self.cells_fw,
                    cells_bw=self.cells_bw,
                    inputs=self.inputs_hidden,
                    initial_states_fw=self.initial_states_fw,
                    initial_states_bw=self.initial_states_bw,
                    dtype=tf.float32,
                    sequence_length=self.input_seq_length)
            else:
                outputs_tuple, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cells_fw[0],
                                                                               cell_bw=self.cells_bw[0],
                                                                               inputs=self.inputs_hidden,
                                                                               sequence_length=self.input_seq_length,
                                                                               initial_state_fw=self.initial_states_fw[0],
                                                                               initial_state_bw=self.initial_states_bw[0],
                                                                               dtype=tf.float32)
                self.outputs = tf.concat(outputs_tuple, 2)
                self.output_state_fw, self.output_state_bw = output_states