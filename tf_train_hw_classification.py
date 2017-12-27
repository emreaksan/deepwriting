import pickle
import sys
import time
import os
import argparse

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

from tf_dataset_hw import *
from tf_data_feeder import *
from tf_models_hw_classification import *
from utils import get_model_dir_timestamp

"""
Model Training Script. 

- Loads model and dataset classes specified in config.
- Creates dataset and data feeder objects for training.
- Creates training model.
- If validation data is provided, creates validation data & data feeder and validation model. Note that validation model
  uses a different computational graph but shares weights with the training model.
- Creates tensorflow routines (i.e., session creation, gradient checks, optimization, summaries, etc.).
- Runs main training loop:
    * Graph ops and summary ops to be evaluated are defined by the model class.
    * Model is evaluated on the full validation data every time. Because of tensorflow input queues, we use an
    unconventional routine. We need to iterate `num_validation_iterations` (# validation samples/batch size) times. 
    Model keeps track of losses and report via `get_validation_summary` method.
"""

def train(config):
    # Fetch Model and Dataset classes.
    Model_cls = getattr(sys.modules[__name__], config['model_cls'])
    Dataset_cls = getattr(sys.modules[__name__], config['dataset_cls'])

    # Training dataset
    training_dataset = Dataset_cls(config['training_data'], use_bow_labels=config.get('use_bow_labels', False), data_augmentation=config.get('data_augmentation', False))
    num_training_iterations = int(training_dataset.num_samples/config['batch_size'])
    print("# training steps per epoch: " + str(num_training_iterations))

    # Create a tensorflow sub-graph that loads batches of samples.
    if config.get('use_bucket_feeder', True) and training_dataset.is_dynamic:
        bucket_edges = training_dataset.get_seq_len_histogram(num_bins=15, collapse_first_and_last_bins=[2,-2])
        data_feeder = DataFeederTF(training_dataset, config['num_epochs'], config['batch_size'], queue_capacity=1024)
        sequence_length, inputs, targets = data_feeder.batch_queue_bucket(bucket_edges,
                                                                          dynamic_pad=training_dataset.is_dynamic,
                                                                          queue_capacity=300,
                                                                          queue_threads=8)
    else:
        data_feeder = DataFeederTF(training_dataset, config['num_epochs'], config['batch_size'], queue_capacity=1024)
        sequence_length, inputs, targets = data_feeder.batch_queue(dynamic_pad=training_dataset.is_dynamic,
                                                                   queue_capacity=512,
                                                                   queue_threads=8)

    if config.get('use_staging_area', False):
        staging_area = TFStagingArea([sequence_length, inputs, targets], device_name="/gpu:0")
        sequence_length, inputs, targets = staging_area.tensors

    # Create training graph.
    with tf.name_scope("training"):
        model = Model_cls(config,
                          reuse=False,
                          input_op=inputs,
                          target_op=targets,
                          input_seq_length_op=sequence_length,
                          input_dims=training_dataset.input_dims,
                          target_dims=training_dataset.target_dims,
                          mode="training")
        model.build_graph()

    # Validation model.
    if config.get('validate_model', False):
        validation_dataset = Dataset_cls(config['validation_data'], use_bow_labels=config.get('use_bow_labels', False), data_augmentation=config.get('data_augmentation', False))
        num_validation_iterations = int(validation_dataset.num_samples/config['batch_size'])
        print("# validation steps per epoch: " + str(num_validation_iterations))
        assert not (num_validation_iterations == 0), "Not enough validation samples."

        valid_data_feeder = DataFeederTF(validation_dataset,
                                         config['num_epochs'],
                                         config['batch_size'],
                                         queue_capacity=1024,
                                         shuffle=False)

        valid_sequence_length, valid_inputs, valid_targets = valid_data_feeder.batch_queue(
                                                                            dynamic_pad=validation_dataset.is_dynamic,
                                                                            queue_capacity=512,
                                                                            queue_threads=4)
        if config.get('use_staging_area', False):
            valid_staging_area = TFStagingArea([valid_sequence_length, valid_inputs, valid_targets], device_name="/gpu:0")
            valid_sequence_length, valid_inputs, valid_targets = valid_staging_area.tensors

        with tf.name_scope("validation"):
            valid_model = Model_cls(config,
                                    reuse=True,
                                    input_op=valid_inputs,
                                    target_op=valid_targets,
                                    input_seq_length_op=valid_sequence_length,
                                    input_dims=validation_dataset.input_dims,
                                    target_dims=validation_dataset.target_dims,
                                    mode="validation")
            valid_model.build_graph()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    # Create step counter (used by optimization routine and learning rate function.)
    global_step = tf.get_variable(name='global_step', trainable=False, initializer=1)

    if config['learning_rate_type'] == 'exponential':
        learning_rate = tf.train.exponential_decay(config['learning_rate'],
                                                   global_step=global_step,
                                                   decay_steps=config['learning_rate_decay_steps'],
                                                   decay_rate=config['learning_rate_decay_rate'],
                                                   staircase=False)
        tf.summary.scalar('training/learning_rate', learning_rate, collections=["training_status"])
    elif config['learning_rate_type'] == 'fixed':
        learning_rate = config['learning_rate']
    else:
        raise Exception("Invalid learning rate type")

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Gradient clipping and a sanity check.
    grads = list(zip(tf.gradients(model.loss, tf.trainable_variables()), tf.trainable_variables()))
    grads_clipped = []
    with tf.name_scope("grad_clipping"):
        for grad, var in grads:
            if grad is not None:
                print(var.name + ": OK")
                if config['grad_clip_by_norm'] > 0:
                    grads_clipped.append((tf.clip_by_norm(grad, config['grad_clip_by_norm']), var))
                elif config['grad_clip_by_value'] > 0:
                    grads_clipped.append(
                        (tf.clip_by_value(grad, -config['grad_clip_by_value'], -config['grad_clip_by_value']), var))
                else:
                    grads_clipped.append((grad, var))
            else:
                print(var.name + ": None")

    train_op = optimizer.apply_gradients(grads_and_vars=grads_clipped, global_step=global_step)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    if config['model_dir']:
        # If model directory already exists, continue training by restoring computation graph.
        # Restore variables.
        if config['checkpoint_id'] is None:
            checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
        else:
            checkpoint_path = os.path.join(config['model_dir'], config_dict['checkpoint_id'])

        print("Continue training with model " + checkpoint_path)
        saver.restore(sess, checkpoint_path)

        step = tf.train.global_step(sess, global_step)
        start_epoch = round(step/(training_dataset.num_samples/config['batch_size']))
    else:
        # Fresh start
        # Create a unique output directory for this experiment.
        config_dict['model_dir'] = get_model_dir_timestamp(base_path=config['model_save_dir'], prefix="tf",
                                                           suffix=config['experiment_name'], connector="-")
        print("Saving to {}\n".format(config['model_dir']))
        start_epoch = 1
        step = 1

    coord = tf.train.Coordinator()
    data_feeder.init(sess, coord)  # TODO (BUG): Enqueue threads must be initialized after definition of train_op.
    if config.get('validate_model', False):
        valid_data_feeder.init(sess, coord)
    queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    queue_threads.append(data_feeder.enqueue_threads)

    # Register and create summary ops.
    summary_dir = os.path.join(config['model_dir'], "summary")
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Create summaries to visualize weights and gradients.
    if config['tensorboard_verbose'] > 1:
        for grad, var in grads:
            tf.summary.histogram(var.name, var, collections=["training_status"])
            tf.summary.histogram(var.name + '/gradient', grad, collections=["training_status"])

    if config['tensorboard_verbose'] > 1:
        tf.summary.scalar("training/queue", math_ops.cast(data_feeder.input_queue.size(), dtypes.float32)*(
        1./data_feeder.queue_capacity), collections=["training_status"])

    # Save configuration.
    pickle.dump(config, open(os.path.join(config['model_dir'], 'config.pkl'), 'wb'))

    # Create lists of training and validation graph operations for session.run. Note that models create them.
    training_summary = tf.summary.merge_all('training_status')
    training_run_ops = [model.loss_summary, training_summary, model.ops_loss, train_op]
    if config.get('validate_model', False):
        validation_run_ops = [valid_model.ops_loss]

    # Fill staging area before getting into main training loop.
    if config['use_staging_area']:
        training_run_ops.append(staging_area.preload_op)
        for i in range(256):
            _ = sess.run(staging_area.preload_op, feed_dict={})

        if config.get('validate_model', False):
            validation_run_ops.append(valid_staging_area.preload_op)
            for i in range(256):
                _ = sess.run(valid_staging_area.preload_op, feed_dict={})
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        for epoch_step in range(num_training_iterations):
            start_time = time.perf_counter()
            step = tf.train.global_step(sess, global_step)

            if (step % config['checkpoint_every_step']) == 0:
                ckpt_save_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
                print("Model saved in file: %s" % ckpt_save_path)

            run_training_output = sess.run(training_run_ops, feed_dict={},)
            summary_writer.add_summary(run_training_output[0], step) # Loss summary
            summary_writer.add_summary(run_training_output[1], step) # Training status summary.

            if step % config['print_every_step'] == 0:
                time_elapsed = (time.perf_counter() - start_time)/config['print_every_step']
                model.log_loss(run_training_output[2], step, epoch, time_elapsed, prefix="TRAIN: ")

            if step % config['validate_every_step'] == 0:
                start_time = time.perf_counter()
                for i in range(num_validation_iterations):
                    run_validation_output = sess.run(validation_run_ops, feed_dict={})
                    valid_model.update_validation_loss(run_validation_output[0])

                valid_summary_feed_dict, valid_eval_loss = valid_model.get_validation_summary()
                valid_summary = sess.run(valid_model.loss_summary, feed_dict=valid_summary_feed_dict)
                summary_writer.add_summary(valid_summary, step)  # Validation loss summary

                time_elapsed = (time.perf_counter() - start_time)/num_validation_iterations
                valid_model.log_loss(valid_eval_loss, step, data_feeder.epoch, time_elapsed, prefix="VALID: ")
                valid_model.reset_validation_loss()

    print("End-of-Training.")
    ckpt_save_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
    print("Model saved in file: %s"%ckpt_save_path)
    print('Model is trained for %d epochs, %d steps.'%(config['num_epochs'], step))

    try:
        sess.run(data_feeder.input_queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(queue_threads, stop_grace_period_secs=5)
    except:
        pass

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--model_save_dir', type=str, default='./runs/', help='path to main model save directory')
    parser.add_argument('-M', '--model_id', dest='model_id',  type=str, help='model folder')
    parser.add_argument('--checkpoint_id', type=str, default=None, help='log and output directory')
    args = parser.parse_args()

    if args.model_id is not None:
        # Restore
        config_dict = pickle.load(open(os.path.join(args.model_save_dir, args.model_id, 'config.pkl'), 'rb'))
        # in case folder is renamed.
        config_dict['model_dir'] = os.path.join(args.model_save_dir, args.model_id)
        config_dict['checkpoint_id'] = args.checkpoint_id
        config_dict['model_id'] = args.model_id
    else:
        # Fresh training
        import config
        config_dict = config.classifier()
        config_dict['model_dir'] = None

    tf.set_random_seed(config_dict['seed'])
    train(config_dict)
