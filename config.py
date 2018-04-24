import tensorflow as tf
import numpy as np

def main():
    config = {}
    config['seed'] = 17
    tf.set_random_seed(17)

    config['training_data'] = './data/dataset_scaled_word_300_eoc_split_old_training.npz'
    config['validation_data'] = './data/dataset_scaled_word_300_eoc_split_old_validation.npz'
    config['validate_model'] = False

    config['model_save_dir'] = './runs/'

    config['checkpoint_every_step'] = 1000
    config['validate_every_step'] = 25 # validation performance
    config['img_summary_every_step'] = 100   # tf_summary
    config['print_every_step'] = 2 # print

    config['reduce_loss'] = "mean_per_step" # "mean" "sum_mean", "mean", "sum".
    config['batch_size'] = 64
    config['num_epochs'] = 200
    config['learning_rate'] = 1e-3
    config['learning_rate_type'] = 'exponential'  # 'fixed'  # 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['learning_rate_decay_rate'] = 0.96

    config['create_timeline'] = False
    config['tensorboard_verbose'] = 0 # 1 for histogram summaries and 2 for latent space norms.
    config['use_dynamic_rnn'] = True
    config['use_bucket_feeder'] = True
    config['use_staging_area'] = True

    config['grad_clip_by_norm'] = 1  # If it is 0, then gradient clipping will not be applied.
    config['grad_clip_by_value'] = 0  # If it is 0, then gradient clipping will not be applied.

    config['vrnn_cell_cls'] = 'HandWritingVRNNGmmCell'
    config['model_cls'] = 'HandwritingVRNNGmmModel'
    config['dataset_cls'] = 'HandWritingDatasetConditionalTF'

    #
    # VRNN Cell settings
    #
    config['output'] = {}
    config['output']['keys'] = ['out_mu', 'out_sigma', 'out_rho', 'out_pen', 'out_eoc']
    config['output']['dims'] = [2, 2, 1, 1, 1] # Ideally these should be set by the model.
    config['output']['activation_funcs'] = [None, 'softplus', 'tanh', 'sigmoid', 'sigmoid']

    config['latent_rnn'] = {}                       # See get_rnn_cell function in tf_model_utils.
    config['latent_rnn']['num_layers'] = 1          # (default: 1)
    config['latent_rnn']['cell_type'] = 'lstm'       # (default: 'lstm')
    config['latent_rnn']['size'] = 512              # (default: 512)

    # Pass None if you want to use fully connected layers in the input or output layers.
    config['input_rnn'] = {}
    if config['input_rnn'] == {}:
        config['input_rnn']['num_layers'] = 1
        config['input_rnn']['cell_type'] = 'lstm'
        config['input_rnn']['size'] = 512

    config['output_rnn'] = None
    if config['output_rnn'] == {}:
        config['output_rnn']['num_layers'] = 1
        config['output_rnn']['cell_type'] = 'lstm'
        config['output_rnn']['size'] = 512

    config['additive_q_mu'] = False
    config['num_fc_layers'] = 1                     # (default: 1)
    config['fc_layer_activation_func'] = 'relu'     # (default: 'relu')
    config['input_keep_prop'] = 1                   # (default: 1)
    config['use_batch_norm_fc'] = False             # (default: False)

    # GMM latent space params.
    config['use_temporal_latent_space'] = True
    config['use_variational_pi'] = True
    config['use_real_pi_labels'] = True
    config['use_pi_as_content'] = False
    config['use_soft_gmm'] = False

    config['use_bow_labels'] = True
    config['pen_threshold'] = 0.4 # Threshold for pen-up event probability.
    config['use_latent_h_in_outputs'] = False  # (default: True)

    config['input_dims'] = None  # Set by the model.
    config['latent_hidden_size'] = 512
    config['latent_size'] = 32

    config['num_gmm_components'] = 70 # We have 70 characters in our alphabet.
    config['gmm_component_size'] = 32

    config['reconstruction_loss'] = "nll_normal"  # "mse", "l1"
    config['loss_weights'] = {'reconstruction_loss': 1, 'kld_loss': 1, 'pen_loss': 1, 'eoc_loss': 1, 'gmm_sigma_regularizer':None, 'classification_loss':1}

    config['experiment_name'] = "deepwriting-synthesis_model"

    return config

def classifier():
    config = {}
    config['seed'] = 17
    tf.set_random_seed(17)

    config['training_data'] = './data/dataset_scaled_word_300_eoc_split_old_training.npz'
    config['validation_data'] = './data/dataset_scaled_word_300_eoc_split_old_validation.npz'
    config['validate_model'] = True

    config['model_save_dir'] = './runs/'

    config['checkpoint_every_step'] = 1000
    config['validate_every_step'] = 100 # validation performance
    config['print_every_step'] = 2 # print

    config['reduce_loss'] = "mean_per_step" #"mean_per_step" "sum_mean", "mean", "sum".
    config['batch_size'] = 64
    config['num_epochs'] = 15
    config['learning_rate'] = 9e-4
    config['learning_rate_type'] = 'exponential'  # 'fixed'  # 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['learning_rate_decay_rate'] = 0.93

    config['tensorboard_verbose'] = 1 # 1 for histogram summaries and 2 for latent space norms.
    config['use_dynamic_rnn'] = True
    config['use_bucket_feeder'] = True
    config['use_staging_area'] = True

    config['grad_clip_by_norm'] = 1  # If it is 0, then gradient clipping will not be applied.
    config['grad_clip_by_value'] = 0  # If it is 0, then gradient clipping will not be applied.

    config['model_cls'] = 'BiDirectionalRNNClassifier' #'RNNClassifier', 'BiDirectionalRNNClassifier
    config['dataset_cls'] = 'HandWritingClassificationDataset'

    config['use_bow_labels'] = True
    config['data_augmentation'] = False

    config['input_layer'] = None
    if config['input_layer'] == {}:
        config['input_layer']['num_layers'] = 1  # number of fully connected (FC) layers on top of RNN.
        config['input_layer']['size'] = 256  # number of FC neurons.
        config['input_layer']['activation_fn'] = 'relu' # type of activation function after each FC layer.

    config['rnn_layer'] = {}  # See get_rnn_cell function in tf_model_utils.
    config['rnn_layer']['num_layers'] = 4  # (default: 1)
    config['rnn_layer']['cell_type'] = 'lstm'  # (default: 'lstm')
    config['rnn_layer']['size'] = 256  # (default: 512)
    config['rnn_layer']['stack_fw_bw_cells'] = True  # (default: True). Only used in bidirectional models.

    config['output_layer'] = {}
    config['output_layer']['num_layers'] = 1 # number of FC layers on top of RNN.
    config['output_layer']['size'] = 256 # number of FC neurons.
    config['output_layer']['activation_fn'] = 'relu'  # type of activation function after each FC layer.
    # Predictions, i.e., outputs of the model.
    config['output_layer'] = {}
    config['output_layer']['out_keys'] = ['out_char', 'out_eoc', 'out_bow']
    config['output_layer']['out_dims'] = None # If set None, then dataset.target_dims will be used.
    config['output_layer']['out_activation_fn'] = [None, 'sigmoid', 'sigmoid']

    config['loss_weights'] = {'classification_loss': 1, 'eoc_loss': 1, 'bow_loss': 1}

    config['experiment_name'] = "deepwriting-classification_model"

    return config
