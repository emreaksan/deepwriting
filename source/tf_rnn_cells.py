import tensorflow as tf

from tf_model_utils import linear, get_activation_fn, get_rnn_cell

"""
VRNN cell classes.

Cell functionality is decomposed into basic methods so that minor variations can be easily implemented by following OOP
paradigm.

`build_training_graph` and `build_sampling_graph` methods are used to create the cell by tensorflow's forward call.
"""

class VRNNCell(tf.contrib.rnn.RNNCell):
    """
    Variational RNN cell.

    Training time behaviour: draws latent vectors from approximate posterior distribution and tries to decrease the
    discrepancy between prior and the approximate posterior distributions.

    Sampling time behaviour: draws latent vectors from the prior distribution to synthesize a sample. This synthetic
    sample is then used to calculate approximate posterior distribution which is fed to RNN to update the state.
    The inputs to the forward call are not used and can be dummy.
    """

    def __init__(self, reuse, mode, config):
        """

        Args:
            config (dict): In addition to standard <key, value> pairs, stores the following dictionaries for rnn and
                output configurations.

                config['output'] = {}
                config['output']['keys']
                config['output']['dims']
                config['output']['activation_funcs']

                config['*_rnn'] = {}
                config['*_rnn']['num_layers'] (default: 1)
                config['*_rnn']['cell_type'] (default: lstm)
                config['*_rnn']['size'] (default: 512)

            reuse: reuse model parameters.
            mode: 'training' or 'sampling'.
        """
        self.input_dims = config['input_dims']
        self.h_dim = config['latent_hidden_size']
        self.z_dim = config['latent_size']
        self.additive_q_mu = config['additive_q_mu']

        self.dropout_keep_prob = config.get('input_keep_prop', 1)
        self.num_linear_layers = config.get('num_fc_layers', 1)
        self.use_latent_h_in_outputs = config.get('use_latent_h_in_outputs', True)
        self.use_batch_norm = config['use_batch_norm_fc']

        self.reuse = reuse
        self.mode = mode
        self.is_sampling = mode == 'sampling'

        if not (mode == "training"):
            self.dropout_keep_prob = 1.0

        self.output_config = config['output']

        self.output_size_ = [self.z_dim]*4
        self.output_size_.extend(self.output_config['dims']) # q_mu, q_sigma, p_mu, p_sigma + model outputs

        self.state_size_ = []
        # Optional. Linear layers will be used if not passed.
        self.input_rnn = False
        if 'input_rnn' in config and not(config['input_rnn'] is None) and len(config['input_rnn'].keys()) > 0:
            self.input_rnn = True
            self.input_rnn_config = config['input_rnn']

            self.input_rnn_cell = get_rnn_cell(scope='input_rnn', **config['input_rnn'])
            self.state_size_.append(self.input_rnn_cell.state_size)

        self.latent_rnn_config = config['latent_rnn']
        self.latent_rnn_cell_type = config['latent_rnn']['cell_type']
        self.latent_rnn_cell = get_rnn_cell(scope='latent_rnn', **config['latent_rnn'])
        self.state_size_.append(self.latent_rnn_cell.state_size)

        # Optional. Linear layers will be used if not passed.
        self.output_rnn = False
        if 'output_rnn' in config and not(config['output_rnn'] is None) and len(config['output_rnn'].keys()) > 0:
            self.output_rnn = True
            self.output_rnn_config = config['output_rnn']

            self.output_rnn_cell = get_rnn_cell(scope='output_rnn', **config['output_rnn'])
            self.state_size_.append(self.output_rnn_cell.state_size)

        self.activation_func = get_activation_fn(config.get('fc_layer_activation_func', 'relu'))
        self.sigma_func = get_activation_fn('softplus')

    @property
    def state_size(self):
        return tuple(self.state_size_)

    @property
    def output_size(self):
        return tuple(self.output_size_)
    #
    # Auxiliary functions
    #
    def draw_sample(self):
        """
        Draws a sample by using cell outputs.

        Returns:

        """
        # Select mu as sample.
        return self.output_components['out_mu']

    def reparametrization(self, mu, sigma, scope):
        """
        Given an isotropic normal distribution (mu and sigma), draws a sample by using reparametrization trick:
        z = mu + sigma*epsilon

        Args:
            mu: mean of isotropic Gaussian distribution.
            sigma: standard deviation of isotropic Gaussian distribution.

        Returns:

        """
        with tf.variable_scope(scope):
            eps = tf.random_normal(sigma.get_shape(), 0.0, 1.0, dtype=tf.float32)
            z = tf.add(mu, tf.multiply(sigma, eps))

            return z

    def phi(self, input_, scope, reuse=None):
        """
        A fully connected layer to increase model capacity and learn and intermediate representation. It is reported to
        be useful in https://arxiv.org/pdf/1506.02216.pdf

        Args:
            input_:
            scope:

        Returns:

        """
        with tf.variable_scope(scope, reuse=reuse):
            phi_hidden = input_
            for i in range(self.num_linear_layers):
                phi_hidden = linear(phi_hidden, self.h_dim, self.activation_func, batch_norm=self.use_batch_norm)

            return phi_hidden

    def latent(self, input_, scope):
        """
        Creates mu and sigma components of a latent distribution. Given an input layer, first applies a fully connected
        layer and then calculates mu & sigma.

        Args:
            input_:
            scope:

        Returns:

        """
        with tf.variable_scope(scope):
            latent_hidden = linear(input_, self.h_dim, self.activation_func, batch_norm=self.use_batch_norm)
            with tf.variable_scope("mu"):
                mu = linear(latent_hidden, self.z_dim)
            with tf.variable_scope("sigma"):
                sigma = linear(latent_hidden, self.z_dim, self.sigma_func)

            return mu, sigma

    def parse_rnn_state(self, state):
        """
        Sets self.latent_h and rnn states.

        Args:
            state:

        Returns:

        """
        latent_rnn_state_idx = 0
        if self.input_rnn is True:
            self.input_rnn_state = state[0]
            latent_rnn_state_idx = 1
        if self.output_rnn is True:
            self.output_rnn_state = state[latent_rnn_state_idx+1]

        self.latent_rnn_state = state[latent_rnn_state_idx]

        if self.latent_rnn_cell_type.lower() == 'gru':
            self.latent_h = self.latent_rnn_state
        else:
            self.latent_h = self.latent_rnn_state.h

    #
    # Functions to build graph.
    #
    def build_training_graph(self, input_, state):
        """

        Args:
            input_:
            state:

        Returns:

        """
        self.parse_rnn_state(state)
        self.input_layer(input_, state)
        self.input_layer_hidden()

        self.latent_p_layer()
        self.latent_q_layer()
        self.phi_z = self.phi_z_q

        self.output_layer_hidden()
        self.output_layer()
        self.update_latent_rnn_layer()

    def build_sampling_graph(self, input_, state):
        self.parse_rnn_state(state)
        self.latent_p_layer()
        self.phi_z = self.phi_z_p

        self.output_layer_hidden()
        self.output_layer()

        # Draw a sample by using predictive distribution.
        synthetic_sample = self.draw_sample()
        self.input_layer(synthetic_sample, state)
        self.input_layer_hidden()
        self.latent_q_layer()
        self.update_latent_rnn_layer()


    def input_layer(self, input_, state):
        """
        Set self.x by applying dropout.
        Args:
            input_:
            state:

        Returns:

        """
        with tf.variable_scope("input"):
            input_components = tf.split(input_, self.input_dims, axis=1)
            self.x = input_components[0]

    def input_layer_hidden(self):
        if self.input_rnn is True:
            self.phi_x_input, self.input_rnn_state = self.input_rnn_cell(self.x, self.input_rnn_state, scope='phi_x_input')
        else:
            self.phi_x_input = self.phi(self.x, scope='phi_x_input')

        if self.dropout_keep_prob < 1.0:
            self.phi_x_input = tf.nn.dropout(self.phi_x_input, keep_prob=self.dropout_keep_prob)

    def latent_q_layer(self):
        input_latent_q = tf.concat((self.phi_x_input, self.latent_h), axis=1)
        if self.additive_q_mu:
            q_mu_delta, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")
            self.q_mu = q_mu_delta + self.p_mu
        else:
            self.q_mu, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")

        q_z = self.reparametrization(self.q_mu, self.q_sigma, scope="z_q")
        self.phi_z_q = self.phi(q_z, scope="phi_z", reuse=True)

    def latent_p_layer(self):
        input_latent_p = tf.concat((self.latent_h), axis=1)
        self.p_mu, self.p_sigma = self.latent(input_latent_p, scope="latent_z_p")

        p_z = self.reparametrization(self.p_mu, self.p_sigma, scope="z_p")
        self.phi_z_p = self.phi(p_z, scope="phi_z")

    def output_layer_hidden(self):
        if self.use_latent_h_in_outputs is True:
            output_layer_hidden = tf.concat((self.phi_z, self.latent_h), axis=1)
        else:
            output_layer_hidden = tf.concat((self.phi_z), axis=1)

        if self.output_rnn is True:
            self.phi_x_output, self.output_rnn_state = self.output_rnn_cell(output_layer_hidden, self.output_rnn_state, scope='phi_x_output')
        else:
            self.phi_x_output = self.phi(output_layer_hidden, scope="phi_x_output")

    def output_layer(self):
        self.output_components = {}
        for key, size, activation_func in zip(self.output_config['keys'], self.output_config['dims'], self.output_config['activation_funcs']):
            with tf.variable_scope(key):
                output_component = linear(self.phi_x_output, size, activation_fn=get_activation_fn(activation_func))
                self.output_components[key] = output_component

    def update_latent_rnn_layer(self):
        input_latent_rnn = tf.concat((self.phi_x_input, self.phi_z), axis=1)
        self.latent_rnn_output, self.latent_rnn_state = self.latent_rnn_cell(input_latent_rnn, self.latent_rnn_state)

    def __call__(self, input_, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            if self.is_sampling:
                self.build_sampling_graph(input_, state)
            else:
                self.build_training_graph(input_, state)

            # Prepare cell output.
            vrnn_cell_output = [self.q_mu, self.q_sigma, self.p_mu, self.p_sigma]
            for key in self.output_config['keys']:
                vrnn_cell_output.append(self.output_components[key])

            # Prepare cell state.
            vrnn_cell_state = []
            if self.input_rnn:
                vrnn_cell_state.append(self.input_rnn_state)

            vrnn_cell_state.append(self.latent_rnn_state)

            if self.output_rnn:
                vrnn_cell_state.append(self.output_rnn_state)

            return tuple(vrnn_cell_output), tuple(vrnn_cell_state)


class VRNNGmmCell(VRNNCell):
    """
    Variational RNN cell with GMM latent space option. See parent class for method documentation. Please note that here
    we use a GMM to learn a continuous representation for categorical inputs. The gradients don't flow through the
    GMM. The model is still trained with classification loss.

    Training time behaviour: draws latent vectors from approximate posterior distribution and tries to decrease the
    discrepancy between prior and the approximate posterior distributions.

    Sampling time behaviour: draws latent vectors from the prior distribution to synthesize a sample. This synthetic
    sample is then used to calculate approximate posterior distribution which is fed to RNN to update the state.
    The inputs to the forward call are not used and can be dummy.
    """
    def __init__(self, reuse, mode, config):
        super(VRNNGmmCell, self).__init__(reuse, mode, config)

        # Latent variable z.
        self.use_temporal_latent_space = config.get('use_temporal_latent_space', True)

        # Latent variable \pi (i.e., content branch).
        # Options for content branch samples: soft_gmm, hard_gmm, pi. (default: hard_gmm)
        self.use_pi_as_content = config.get('use_pi_as_content', False) # If True, label probabilities (\pi) or one-hot-encoded labels will be used as content by bypassing GMM.
        self.is_gmm_active = not(self.use_pi_as_content)
        self.use_soft_gmm = config.get('use_soft_gmm', False)

        self.use_real_pi_labels = config.get('use_real_pi_labels', False) # Ground truth content label is given.
        self.use_variational_pi = config.get('use_variational_pi', False)

        assert (not self.use_pi_as_content) or (self.use_pi_as_content and self.use_real_pi_labels), "`use_real_pi_labels` must be True if `use_pi_as_content` is True."
        assert self.use_variational_pi or self.use_temporal_latent_space, "Both `use_temporal_latent_space` and `use_variational_pi` can't be False."
        assert not (self.use_soft_gmm and self.use_real_pi_labels), "Both `use_soft_gmm` and `use_real_pi_labels` are True."
        assert not (self.use_real_pi_labels and len(self.input_dims) < 2), "Class labels are missing for pi_labels."

        if (self.use_soft_gmm is False) and (self.use_temporal_latent_space is False):
            print("Warning: there is no differentiable latent space component.")
        if (self.use_variational_pi is False) and (self.use_real_pi_labels is False):
            print("Warning: there is no information source for GMM components.")

        self.num_gmm_components = config['num_gmm_components']
        self.gmm_component_size = config['gmm_component_size']

        if self.is_gmm_active:
            # Create mean and sigma variables of gmm components.
            with tf.variable_scope("gmm_latent", reuse=self.reuse):
                self.gmm_mu_vars = tf.get_variable("mu", dtype=tf.float32, initializer=tf.random_uniform([self.num_gmm_components, self.gmm_component_size], -1.0, 1.0))
                self.gmm_sigma_vars = self.sigma_func(tf.get_variable("sigma", dtype=tf.float32, initializer=tf.constant_initializer(1), shape=[self.num_gmm_components, self.gmm_component_size]))
        else:
            self.gmm_mu_vars = self.gmm_sigma_vars = None
            self.gmm_component_size = self.input_dims[1]

        self.output_size_ = []
        if self.use_temporal_latent_space:
            self.output_size_.extend([self.z_dim]*4) # q_mu, q_sigma, p_mu, p_sigma

        self.output_size_.append(self.gmm_component_size)  # z_gmm
        self.output_size_.append(self.num_gmm_components)  # q_pi
        if self.use_variational_pi:
            self.output_size_.append(self.num_gmm_components)  # p_pi

        self.output_size_.extend(self.output_config['dims'])  # model outputs

    def get_gmm_components(self):
        return self.gmm_mu_vars, self.gmm_sigma_vars

    @property
    def state_size(self):
        return tuple(self.state_size_)

    @property
    def output_size(self):
        return tuple(self.output_size_)

    #
    # Functions to build graph.
    #
    def build_training_graph(self, input_, state):
        self.parse_rnn_state(state)
        self.input_layer(input_, state)
        self.input_layer_hidden()

        if self.use_temporal_latent_space:
            self.latent_p_layer()
            self.latent_q_layer()
            self.phi_z = self.phi_z_q # Use approximate distribution in training mode.

        if self.use_variational_pi:
            self.latent_p_pi()

        self.latent_q_pi()

        self.gmm_pi = self.logits_q_pi
        if self.use_real_pi_labels:
            self.gmm_pi = self.real_pi

        self.latent_gmm()

        self.output_layer_hidden()
        self.output_layer()
        self.update_latent_rnn_layer()


    def build_sampling_graph(self, input_, state):
        self.parse_rnn_state(state)
        if self.use_real_pi_labels: # Labels are fed for sampling.
            self.input_layer(input_, state)

        if self.use_temporal_latent_space:
            self.latent_p_layer()
            self.phi_z = self.phi_z_p

        if self.use_variational_pi:
            self.latent_p_pi()
            self.gmm_pi = self.logits_p_pi

        if self.use_real_pi_labels:
            self.gmm_pi = self.real_pi

        self.latent_gmm()

        self.output_layer_hidden()
        self.output_layer()

        # Draw a sample by using predictive distribution.
        synthetic_sample = self.draw_sample() # This will update self.x
        self.input_layer(synthetic_sample, state)
        self.input_layer_hidden()

        if self.use_temporal_latent_space:
            self.latent_q_layer()

        self.latent_q_pi()

        self.update_latent_rnn_layer()


    def latent_gmm(self):
        if self.use_pi_as_content:
            self.gmm_z = self.gmm_pi

        elif self.use_soft_gmm:
            with tf.name_scope("latent_z_gmm"):
                eps = tf.random_normal((self.gmm_pi.get_shape().as_list()[0], self.num_gmm_components, self.gmm_component_size), 0.0, 1.0, dtype=tf.float32)
                z = tf.add(self.gmm_mu_vars, tf.multiply(self.gmm_sigma_vars, eps))

                gmm_pi = tf.expand_dims(self.gmm_pi, axis=1)
                # [batch, 1, num_components] x [batch, num_components, component_size] -> [batch, 1, component_size]
                self.gmm_z = tf.squeeze(tf.matmul(gmm_pi, z), axis=1)
        else:
            with tf.name_scope("latent_z_gmm"):
                mixture_components = tf.expand_dims(tf.argmax(self.gmm_pi, axis=-1), axis=-1)
                gmm_mu = tf.gather_nd(self.gmm_mu_vars, mixture_components)
                gmm_sigma = tf.gather_nd(self.gmm_sigma_vars, mixture_components)
            # z = mu + sigma*epsilon
            self.gmm_z = self.reparametrization(gmm_mu, gmm_sigma, scope="latent_z_gmm")

        self.phi_z_gmm = self.phi(self.gmm_z, scope="phi_z_gmm")


    def latent_q_pi(self):
        input_ = tf.concat((self.x, self.latent_h), axis=1)
        with tf.variable_scope("latent_q_pi"):
            phi_pi = linear(input_, self.h_dim, self.activation_func, batch_norm=self.use_batch_norm)
            self.logits_q_pi = linear(phi_pi, self.num_gmm_components, activation_fn=None,
                                      batch_norm=self.use_batch_norm)


    def latent_p_pi(self):
        input_ = tf.concat((self.latent_h), axis=1)
        with tf.variable_scope("latent_p_pi"):
            phi_pi = linear(input_, self.h_dim, self.activation_func, batch_norm=self.use_batch_norm)
            self.logits_p_pi = linear(phi_pi, self.num_gmm_components, activation_fn=None,
                                      batch_norm=self.use_batch_norm)

    def latent_p_layer(self):
        input_latent_p = tf.concat((self.latent_h), axis=1)
        self.p_mu, self.p_sigma = self.latent(input_latent_p, scope="latent_z_p")

        p_z = self.reparametrization(self.p_mu, self.p_sigma, scope="z_p")
        self.phi_z_p = self.phi(p_z, scope="phi_z")

    def latent_q_layer(self):
        input_latent_q = tf.concat((self.phi_x_input, self.latent_h), axis=1)
        if self.additive_q_mu:
            q_mu_delta, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")
            self.q_mu = q_mu_delta + self.p_mu
        else:
            self.q_mu, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")

        q_z = self.reparametrization(self.q_mu, self.q_sigma, scope="z_q")
        self.phi_z_q = self.phi(q_z, scope="phi_z", reuse=True)


    def input_layer(self, input_, state):
        with tf.variable_scope("input"):
            input_components = tf.split(input_, self.input_dims, axis=1)
            #self.x = tf.nn.dropout(input_components[0], keep_prob=self.dropout_keep_prob)
            self.x = input_components[0]

            if self.use_real_pi_labels:
                self.real_pi = input_components[1]

    def input_layer_hidden(self):
        if self.input_rnn is True:
            self.phi_x_input, self.input_rnn_state = self.input_rnn_cell(self.x, self.input_rnn_state, scope='phi_x_input')
        else:
            self.phi_x_input = self.phi(self.x, scope='phi_x_input')

        if self.dropout_keep_prob < 1.0:
            self.phi_x_input = tf.nn.dropout(self.phi_x_input, keep_prob=self.dropout_keep_prob)

    def output_layer_hidden(self):
        input_list = [self.phi_z_gmm]
        if self.use_temporal_latent_space:
            input_list.append(self.phi_z)

        if self.use_latent_h_in_outputs is True:
            input_list.append(self.latent_h)

        inputs_ = tf.concat(input_list, axis=1)

        if self.output_rnn is True:
            self.phi_x_output, self.output_rnn_state = self.output_rnn_cell(inputs_, self.output_rnn_state, scope='phi_x_output')
        else:
            self.phi_x_output = self.phi(inputs_, scope="phi_x_output")

    def output_layer(self):
        self.output_components = {}
        for key, size, activation_func in zip(self.output_config['keys'], self.output_config['dims'],
                                              self.output_config['activation_funcs']):
            with tf.variable_scope(key):
                output_component = linear(self.phi_x_output, size, activation_fn=get_activation_fn(activation_func))
                self.output_components[key] = output_component

    def update_latent_rnn_layer(self):
        input_list = [self.phi_x_input, self.phi_z_gmm]

        if self.use_temporal_latent_space:
            input_list.append(self.phi_z)

        input_latent_rnn = tf.concat(input_list, axis=1)
        self.latent_rnn_output, self.latent_rnn_state = self.latent_rnn_cell(input_latent_rnn, self.latent_rnn_state)


    def __call__(self, input_, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            if self.is_sampling:
                self.build_sampling_graph(input_, state)
            else:
                self.build_training_graph(input_, state)

            # Prepare cell output.
            vrnn_cell_output = []
            if self.use_temporal_latent_space:
                vrnn_cell_output = [self.q_mu, self.q_sigma, self.p_mu, self.p_sigma]

            vrnn_cell_output.append(self.gmm_z)
            vrnn_cell_output.append(self.logits_q_pi)
            if self.use_variational_pi:
                vrnn_cell_output.append(self.logits_p_pi)

            for key in self.output_config['keys']:
                vrnn_cell_output.append(self.output_components[key])

            # Prepare cell state.
            vrnn_cell_state = []
            if self.input_rnn:
                vrnn_cell_state.append(self.input_rnn_state)

            vrnn_cell_state.append(self.latent_rnn_state)

            if self.output_rnn:
                vrnn_cell_state.append(self.output_rnn_state)

            return tuple(vrnn_cell_output), tuple(vrnn_cell_state)


class HandWritingVRNNGmmCell(VRNNGmmCell):
    """
    Variational RNN cell for modeling of digital handwriting.

    Training time behaviour: draws latent vectors from approximate posterior distribution and tries to decrease the
    discrepancy between prior and the approximate posterior distributions.

    Inference time behaviour: draws latent vectors from the prior distribution to synthesize a sample. This synthetic
    sample is then used to calculate approximate posterior distribution which is fed to RNN to update the state.
    The inputs to the forward call are not used and can be dummy.
    """

    def __init__(self, reuse, mode, config):
        super(HandWritingVRNNGmmCell, self).__init__(reuse, mode, config)

        # Use beginning-of-word (bow) labels as input,
        self.use_bow_labels = config.get('use_bow_labels', False)
        self.pen_threshold = config.get('pen_threshold', 0.4)

    # Auxiliary functions.
    def binarize(self, input_):
        """
        Transforms continuous values in [0,1] to {0,1} by applying a step function.

        Args:
            cont: tensor with continuous data in [0,1].

        Returns:

        """
        return tf.where(tf.greater(input_, tf.fill(tf.shape(input_), self.pen_threshold)),
                        tf.fill(tf.shape(input_), 1.0), tf.fill(tf.shape(input_), 0.0))

    def draw_sample(self):
        # Select mu as sample.
        sample_components = [self.output_components['out_mu'], self.binarize(self.output_components['out_pen'])]
        if self.use_real_pi_labels:
            sample_components.append(self.gmm_pi)
        if self.use_bow_labels:
            sample_components.append(self.bow_labels)
        return tf.concat(sample_components, axis=1)

    def input_layer(self, input_, state):
        with tf.variable_scope("input"):

            input_components = tf.split(input_, self.input_dims, axis=1)
            self.x = tf.nn.dropout(input_components[0], keep_prob=self.dropout_keep_prob)

            if self.use_real_pi_labels:
                self.real_pi = input_components[1] # Character labels.

            if self.use_bow_labels:
                self.bow_labels = input_components[2]

    def latent_p_layer(self):
        input_latent_p_list = [self.latent_h]
        input_latent_p = tf.concat(input_latent_p_list, axis=1)
        self.p_mu, self.p_sigma = self.latent(input_latent_p, scope="latent_z_p")

        p_z = self.reparametrization(self.p_mu, self.p_sigma, scope="z_p")
        self.phi_z_p = self.phi(p_z, scope="phi_z")

    def latent_q_layer(self):
        input_latent_p_list = [self.phi_x_input, self.latent_h]
        input_latent_q = tf.concat(input_latent_p_list, axis=1)

        if self.additive_q_mu:
            q_mu_delta, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")
            self.q_mu = q_mu_delta + self.p_mu
        else:
            self.q_mu, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")

        q_z = self.reparametrization(self.q_mu, self.q_sigma, scope="z_q")
        self.phi_z_q = self.phi(q_z, scope="phi_z", reuse=True)

    def output_layer_hidden(self):
        input_list = [self.phi_z_gmm]
        if self.use_temporal_latent_space:
            input_list.append(self.phi_z)

        if self.use_latent_h_in_outputs is True:
            input_list.append(self.latent_h)

        if self.use_bow_labels:
            input_list.append(self.bow_labels)

        inputs_ = tf.concat(input_list, axis=1)

        if self.output_rnn is True:
            self.phi_x_output, self.output_rnn_state = self.output_rnn_cell(inputs_,
                                                                            self.output_rnn_state,
                                                                            scope='phi_x_output')
        else:
            self.phi_x_output = self.phi(inputs_, scope="phi_x_output")
