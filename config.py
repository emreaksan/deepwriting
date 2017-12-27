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
    config['img_summary_every_step'] = 10   # tf_summary
    config['print_every_step'] = 2 # print

    config['reduce_loss'] = "mean_per_step" # "mean" "sum_mean", "mean", "sum".
    config['batch_size'] = 64
    config['num_epochs'] = 200
    config['learning_rate'] = 1e-3
    config['learning_rate_type'] = 'exponential'  # 'fixed'  # 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['learning_rate_decay_rate'] = 0.96

    config['create_timeline'] = False
    config['tensorboard_verbose'] = 1 # 1 for histogram summaries and 2 for latent space norms.
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
    config['use_latent_h_in_outputs'] = False       # (default: True)
    config['use_batch_norm_fc'] = False             # (default: False)

    # GMM latent space params.
    config['use_temporal_latent_space'] = True
    config['use_variational_pi'] = True
    config['use_real_pi_labels'] = True
    config['use_pi_as_content'] = False
    config['use_soft_gmm'] = False
    config['use_bow_labels'] = True

    config['input_dims'] = None  # Set by the model.
    config['latent_hidden_size'] = 512
    config['latent_size'] = 64

    config['num_gmm_components'] = 70
    config['gmm_component_size'] = 32

    config['reconstruction_loss'] = "nll_normal"  # "nll_normal_iso", "mse", "l2", "l1"
    config['loss_weights'] = {'reconstruction_loss': 1, 'kld_loss': 1, 'pen_loss': 1, 'eoc_loss': 1, 'gmm_sigma_regularizer':None, 'classification_loss':1}
    config['loss_weights']['kld_loss'] = np.int32(np.linspace(0, 15, 10)) # Start from 0 and increase until 1 in 10 steps until epoch 5.

    config['experiment_name'] = "lstm2-512_64_32-relu-class_loss-no_sigma_reg-var_pi-bow-no_latent_h"

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

    config['reduce_loss'] = "mean_per_step" # "mean" "sum_mean", "mean", "sum".
    config['batch_size'] = 64
    config['num_epochs'] = 150
    config['learning_rate'] = 9e-4
    config['learning_rate_type'] = 'exponential'  # 'fixed'  # 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['learning_rate_decay_rate'] = 0.95

    config['tensorboard_verbose'] = 1 # 1 for histogram summaries and 2 for latent space norms.
    config['use_dynamic_rnn'] = True
    config['use_bucket_feeder'] = True
    config['use_staging_area'] = True

    config['grad_clip_by_norm'] = 1  # If it is 0, then gradient clipping will not be applied.
    config['grad_clip_by_value'] = 0  # If it is 0, then gradient clipping will not be applied.

    config['model_cls'] = 'BiDirectionalRNNClassifier' #'RNNClassifier', 'BiDirectionalRNNClassifier
    config['dataset_cls'] = 'HandWritingClassificationDataset'

    config['use_bow_labels'] = True
    config['data_augmentation'] = True

    config['input_layer'] = {}
    config['input_layer']['num_layers'] = 1  # number of fully connected (FC) layers on top of RNN.
    config['input_layer']['size'] = 256  # number of FC neurons.
    config['input_layer']['activation_fn'] = 'relu' # type of activation function after each FC layer.

    config['rnn_layer'] = {}  # See get_rnn_cell function in tf_model_utils.
    config['rnn_layer']['num_layers'] = 3  # (default: 1)
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

    config['experiment_name'] = "classification-fc1_256-bi_stacked_lstm2_256-fc1_256-relu-augment"

    return config
