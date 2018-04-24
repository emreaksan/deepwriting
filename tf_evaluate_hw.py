import tensorflow as tf
import numpy as np

import sys
import os
import argparse
import json
from scipy.misc import imsave

from tf_dataset_hw import *
from tf_models import VRNNGMM
from tf_models_hw import HandwritingVRNNGmmModel, HandwritingVRNNModel
from  utils_visualization import plot_latent_variables, plot_latent_categorical_variables, plot_matrix_and_get_image, plot_and_get_image
import visualize_hw as visualize

# Sampling options
run_gmm_eval = False
run_original_sample = False
run_reconstruction = False
run_biased_sampling = True
run_unbiased_sampling = True
run_colored_png_output = False

# Sampling hyper-parameters
eoc_threshold = 0.1
cursive_threshold = 0.005
keep_style_states = [0, 1, 0] # input, latent, output rnn cell states.
ref_len = None # Use the whole sequence.
seq_len = 800
gmm_num_samples = 500 # For run_gmm_eval only.
#conditional_texts = ["monopoly of lead", "how in caves a", "interests of a moder"]
conditional_texts = ["I am a synthetic sample", "I can write this line in so many styles."]
reference_sample_ids = [107, 226, 696]

# Sampling output options
plot_eoc = True
plot_latent_norm = False
plot_latent_vars = False
save_plots = True
show_plots = False

def plot_eval_details(data_dict, sample, save_dir, save_name):
    visualize.draw_stroke_svg(sample, factor=0.001, svg_filename=os.path.join(save_dir, save_name + '.svg'))

    plot_data = {}
    if run_colored_png_output:
        synthetic_eoc = np.squeeze(data_dict['out_eoc'])
        visualize.draw_stroke_cv2_colored(sample, factor=0.001, color_labels=synthetic_eoc > eoc_threshold,
                                          black_zero=False, output_path=os.path.join(save_dir, save_name + 'colored.png'))

    if plot_latent_vars and 'p_mu' in data_dict:
        plot_data['p_mu'] = np.transpose(data_dict['p_mu'][0], [1, 0])
        plot_data['q_mu'] = np.transpose(data_dict['q_mu'][0], [1, 0])
        plot_data['q_sigma'] = np.transpose(data_dict['q_sigma'][0], [1, 0])
        plot_data['p_sigma'] = np.transpose(data_dict['p_sigma'][0], [1, 0])

        plot_img = plot_latent_variables(plot_data, show_plot=show_plots)
        if save_plots:
            imsave(os.path.join(save_dir, save_name + '_normal.png'), plot_img)

    if plot_latent_vars and 'p_pi' in data_dict:
        plot_data['p_pi'] = np.transpose(data_dict['p_pi'][0], [1, 0])
        plot_data['q_pi'] = np.transpose(data_dict['q_pi'][0], [1, 0])
        plot_img = plot_latent_categorical_variables(plot_data, show_plot=show_plots)
        if save_plots:
            imsave(os.path.join(save_dir, save_name + '_pi.png'), plot_img)

    if plot_eoc and 'out_eoc' in data_dict:
        plot_img = plot_and_get_image(np.squeeze(data_dict['out_eoc']))
        imsave(os.path.join(save_dir, save_name + '_eoc.png'), plot_img)

    # Same for every sample.
    if 'gmm_mu' in data_dict:
        gmm_mu_img = plot_matrix_and_get_image(data_dict['gmm_mu'])
        gmm_sigma_img = plot_matrix_and_get_image(data_dict['gmm_sigma'])
        if save_plots:
            imsave(os.path.join(save_dir, 'gmm_mu.png'), gmm_mu_img)
            imsave(os.path.join(save_dir, 'gmm_sigma.png'), gmm_sigma_img)

    return plot_data

def do_evaluation(config, qualitative_analysis=True, quantitative_analysis=True, verbose=0):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    Model_cls = getattr(sys.modules[__name__], config['model_cls'])
    Dataset_cls = getattr(sys.modules[__name__], config['dataset_cls'])

    batch_size = 1
    data_sequence_length = None
    # Load validation dataset to fetch statistics.
    if issubclass(Dataset_cls, HandWritingDatasetConditional):
        validation_dataset = Dataset_cls(config['validation_data'], var_len_seq=True, use_bow_labels=config['use_bow_labels'])
    elif issubclass(Dataset_cls, HandWritingDataset):
        validation_dataset = Dataset_cls(config['validation_data'], var_len_seq=True)
    else:
        raise("Unknown dataset class.")

    strokes = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(validation_dataset.input_dims)])
    targets = tf.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(validation_dataset.target_dims)])
    sequence_length = tf.placeholder(tf.int32, shape=[batch_size])

    # Create inference graph.
    with tf.name_scope("validation"):
        inference_model = Model_cls(config,
                                    reuse=False,
                                    input_op=strokes,
                                    target_op=targets,
                                    input_seq_length_op=sequence_length,
                                    input_dims=validation_dataset.input_dims,
                                    target_dims=validation_dataset.target_dims,
                                    batch_size=batch_size,
                                    mode="validation",
                                    data_processor=validation_dataset)
        inference_model.build_graph()
        inference_model.create_image_summary(validation_dataset.prepare_for_visualization)

    # Create sampling graph.
    with tf.name_scope("sampling"):
        model = Model_cls(config,
                          reuse=True,
                          input_op=strokes,
                          target_op=None,
                          input_seq_length_op=sequence_length,
                          input_dims=validation_dataset.input_dims,
                          target_dims=validation_dataset.target_dims,
                          batch_size=batch_size,
                          mode="sampling",
                          data_processor=validation_dataset)
        model.build_graph()

    # Create a session object and initialize parameters.
    sess = tf.Session()
    # Restore computation graph.
    try:
        saver = tf.train.Saver()
        # Restore variables.
        if config['checkpoint_id'] is None:
            checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
        else:
            checkpoint_path = os.path.join(config['model_dir'], config['checkpoint_id'])

        print("Loading model " + checkpoint_path)
        saver.restore(sess, checkpoint_path)
    except:
        raise Exception("Model is not found.")

    if run_gmm_eval:
        from sklearn import manifold
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        gmm_mus, gmm_sigmas = model.evaluate_gmm_latent_space(sess)

        gmm_component_ids = [2,3,11,12,13,14,15,39,40]
        gmm_legend_labels = ["1", "2", "a", "b", "c", "d", "e", "C", "D"]
        num_components = len(gmm_component_ids) #gmm_mus.shape[0]
        size_components = gmm_mus.shape[1]

        gmm_samples = np.zeros((num_components*gmm_num_samples,size_components))
        gmm_sample_labels = np.zeros(num_components*gmm_num_samples)

        for comp_idx in range(num_components):
            epsilon = np.random.normal(0, 1, (gmm_num_samples, gmm_mus.shape[1]))
            gmm_samples[comp_idx*gmm_num_samples:comp_idx*gmm_num_samples+gmm_num_samples,: ] = gmm_mus[comp_idx]+gmm_sigmas[comp_idx]*epsilon
            gmm_sample_labels[comp_idx*gmm_num_samples:comp_idx*gmm_num_samples+gmm_num_samples] = np.ones(gmm_num_samples)*comp_idx

        # Creating a discrete colorbar
        colors = plt.cm.jet(np.linspace(0, 1, num_components))

        Y = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(gmm_samples)
        #Y = decomposition.TruncatedSVD(n_components=2).fit_transform(gmm_samples)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)


        current_plot_range = 0
        previous_plot_range = 0
        for i, c in enumerate(colors):
            previous_plot_range += current_plot_range
            current_plot_range = gmm_sample_labels[gmm_sample_labels == i].size
            plt.scatter(Y[previous_plot_range:previous_plot_range+current_plot_range, 0],
                Y[previous_plot_range:previous_plot_range+current_plot_range, 1],
                20, lw=.25, marker='o', color=c, label=gmm_legend_labels[i], alpha=0.9, antialiased=True,
                zorder=3)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.legend()
        plt.axis('tight')
        plt.show()

    keyword_args = {}
    keyword_args['conditional_inputs'] = None
    keyword_args['eoc_threshold'] = eoc_threshold
    keyword_args['cursive_threshold'] = cursive_threshold
    keyword_args['use_sample_mean'] = True

    if quantitative_analysis:
        pass

    if qualitative_analysis:
        for real_img_idx in reference_sample_ids:
            _, stroke_model_input, _ = validation_dataset.fetch_sample(real_img_idx)
            stroke_sample = stroke_model_input[:, :, 0:3]

            if run_reconstruction or run_biased_sampling:
                inference_results = inference_model.reconstruct_given_sample(session=sess, inputs=stroke_model_input)

            if run_original_sample:
                svg_path = os.path.join(config['eval_dir'], "real_image_"+str(real_img_idx)+'.svg')
                visualize.draw_stroke_svg(validation_dataset.undo_normalization(validation_dataset.samples[real_img_idx], detrend_sample=False), factor=0.001, svg_filename=svg_path)

            if run_reconstruction:
                svg_path = os.path.join(config['eval_dir'], "reconstructed_image_" + str(real_img_idx) + '.svg')
                visualize.draw_stroke_svg(validation_dataset.undo_normalization(inference_results[0]['output_sample'][0], detrend_sample=False), factor=0.001, svg_filename=svg_path)

            # Conditional handwriting synthesis.
            for text_id, text in enumerate(conditional_texts):
                keyword_args['conditional_inputs'] = text
                if config.get('use_real_pi_labels', False) and isinstance(model, VRNNGMM):
                    if run_biased_sampling:
                        biased_sampling_results = model.sample_biased(session=sess, seq_len=seq_len,
                                                                      prev_state=inference_results[0]['state'],
                                                                      prev_sample=stroke_sample,
                                                                      **keyword_args)

                        save_name = 'synthetic_biased_ref(' + str(real_img_idx) + ')_(' + str(text_id) + ')'
                        synthetic_sample = validation_dataset.undo_normalization(biased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                        if save_plots:
                            plot_eval_details(biased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

                        # Without beautification: set False
                        # Apply beautification: set True.
                        keyword_args['use_sample_mean'] = True
                        biased_sampling_results = model.sample_biased(session=sess, seq_len=seq_len,
                                                                      prev_state=inference_results[0]['state'],
                                                                      prev_sample=stroke_sample,
                                                                      **keyword_args)

                        save_name = 'synthetic_biased_sampled_ref(' + str(real_img_idx) + ')_(' + str(text_id) + ')'
                        synthetic_sample = validation_dataset.undo_normalization(biased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                        if save_plots:
                            plot_eval_details(biased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

                    if run_unbiased_sampling:
                        unbiased_sampling_results = model.sample_unbiased(session=sess, seq_len=seq_len, **keyword_args)

                        save_name = 'synthetic_unbiased_(' + str(text_id) + ')'
                        synthetic_sample = validation_dataset.undo_normalization(unbiased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                        if save_plots:
                            plot_eval_details(unbiased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

                        # Without beautification.
                        keyword_args['use_sample_mean'] = True
                        unbiased_sampling_results = model.sample_unbiased(session=sess, seq_len=seq_len, **keyword_args)
                        save_name = 'synthetic_unbiased_sampled(' + str(text_id) + ')'
                        synthetic_sample = validation_dataset.undo_normalization(unbiased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                        if save_plots:
                            plot_eval_details(unbiased_sampling_results[0], synthetic_sample, config['eval_dir'],save_name)

                else:
                    if run_biased_sampling:
                        biased_sampling_results = model.sample_biased(session=sess, seq_len=seq_len,
                                                                      prev_state=inference_results[0]['state'],
                                                                      prev_sample=stroke_sample)

                        save_name = 'synthetic_biased_ref(' + str(real_img_idx) + ')_(' + str(text_id) + ')'
                        synthetic_sample = validation_dataset.undo_normalization(biased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                        if save_plots:
                            plot_eval_details(biased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

                    if run_unbiased_sampling:
                        unbiased_sampling_results = model.sample_unbiased(session=sess, seq_len=seq_len)

                        save_name = 'synthetic_unbiased_(' + str(text_id) + ')'
                        synthetic_sample = validation_dataset.undo_normalization(unbiased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                        if save_plots:
                            plot_eval_details(unbiased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-S', '--model_save_dir', dest='model_save_dir', type=str, default='./runs/', help='path to main model save directory')
    parser.add_argument('-E', '--eval_dir', type=str, default='./runs_evaluation/', help='path to main log/output directory')
    parser.add_argument('-M', '--model_id', dest='model_id', type=str, help='model folder')
    parser.add_argument('-C', '--checkpoint_id', type=str, default=None, help='log and output directory')
    parser.add_argument('-QN', '--quantitative', dest='quantitative', action="store_true", help='Run quantitative analysis')
    parser.add_argument('-QL', '--qualitative', dest='qualitative', action="store_true", help='Run qualitative analysis')
    parser.add_argument('-V', '--verbose', dest='verbose', type=int, default=1, help='Verbosity')
    args = parser.parse_args()

    #config_dict = pickle.load(open(os.path.join(args.model_save_dir, args.model_id, 'config.pkl'), 'rb'))
    config_dict = json.load(open(os.path.abspath(os.path.join(args.model_save_dir, args.model_id, 'config.json')), 'r'))
    # in case folder is renamed.
    config_dict['model_dir'] = os.path.join(args.model_save_dir, args.model_id)
    config_dict['checkpoint_id'] = args.checkpoint_id

    if args.eval_dir is None:
        config_dict['eval_dir'] = config_dict['model_dir']
    else:
        config_dict['eval_dir'] = os.path.join(args.eval_dir, args.model_id)

    if not os.path.exists(config_dict['eval_dir']):
        os.makedirs(config_dict['eval_dir'])

    do_evaluation(config_dict, quantitative_analysis=args.quantitative, qualitative_analysis=args.qualitative, verbose=args.verbose)