#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import logging
import string
import random
from datetime import datetime

from dimenet.model.dimenet import DimeNet
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

from sacred import Experiment

from seml import database_utils as db_utils
from seml import misc

ex = Experiment()
misc.setup_logger(ex)

# TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARN')
tf.autograph.set_verbosity(1)


@ex.config
def config():
    overwrite = None
    db_collection = None
    ex.observers.append(db_utils.create_mongodb_observer(
        db_collection, overwrite=overwrite))


@ex.automain
def run(num_features, num_blocks, num_bilinear, num_spherical, num_radial,
        num_before_skip, num_after_skip, num_dense_output,
        cutoff, envelope_exponent, dataset, num_train, num_valid,
        data_seed, max_steps, learning_rate, ema_decay,
        decay_steps, warmup_steps, decay_rate, batch_size,
        summary_interval, validation_interval, save_interval, restart, targets,
        comment, logdir):

    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
    def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    # Create directories
    # A unique directory name is created for this run based on the input
    if restart is None:
        directory = (logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id_generator()
                     + "_" + os.path.basename(dataset)
                     + "_f" + str(num_features)
                     + "_bi" + str(num_bilinear)
                     + "_sbf" + str(num_spherical)
                     + "_rbf" + str(num_radial)
                     + "_b" + str(num_blocks)
                     + "_nbs" + str(num_before_skip)
                     + "_nas" + str(num_after_skip)
                     + "_no" + str(num_dense_output)
                     + "_cut" + str(cutoff)
                     + "_env" + str(envelope_exponent)
                     + f"_lr{learning_rate:.2e}"
                     + f"_dec{decay_steps:.2e}"
                     + "_" + '-'.join(targets)
                     + "_" + comment)
    else:
        directory = restart

    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_dir = os.path.join(directory, 'best')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    best_loss_file = os.path.join(best_dir, 'best_loss.npz')
    best_ckpt_folder = best_dir
    step_ckpt_folder = log_dir

    def create_summary(dictionary):
        """Create a summary from key-value pairs given a dictionary"""
        for key, value in dictionary.items():
            tf.summary.scalar(key, value)

    # Initialize summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)

    with summary_writer.as_default():
        logging.info("Load dataset")
        data_container = DataContainer(dataset, cutoff=cutoff, target_keys=targets)

        # Initialize DataProvider (splits dataset into training, validation and test set based on data_seed)
        data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                                     seed=data_seed, randomized=True)
        train = {}
        validation = {}

        # Initialize datasets
        train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
        train['dataset_iter'] = iter(train['dataset'])
        validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
        validation['dataset_iter'] = iter(validation['dataset'])

        logging.info("Initialize model")
        model = DimeNet(num_features=num_features, num_blocks=num_blocks, num_bilinear=num_bilinear,
                        num_spherical=num_spherical, num_radial=num_radial,
                        cutoff=cutoff, envelope_exponent=envelope_exponent,
                        num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                        num_dense_output=num_dense_output, num_targets=len(targets),
                        activation=swish)

        logging.info("Prepare training")
        # Save/load best recorded loss (only the best model is saved)
        save_keys = ['step', 'loss', 'mean_mae', 'mean_log_mae', *targets]
        best_res = {}
        if os.path.isfile(best_loss_file):
            loss_file = np.load(best_loss_file)
            for key in save_keys:
                best_res[key] = loss_file[key].item()
        else:
            for key in save_keys[1:]:
                best_res[key] = np.inf
            best_res['step'] = 0
            np.savez(best_loss_file, **best_res)

        # Initialize trainer
        trainer = Trainer(model, learning_rate, warmup_steps,
                          decay_steps, decay_rate,
                          ema_decay=ema_decay, max_grad_norm=1000)

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

        # Restore latest checkpoint
        ckpt_restored = tf.train.latest_checkpoint(log_dir)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored)

        def calculate_mae(targets, preds):
            """Calculate mean absolute error between two values."""
            delta = tf.abs(targets - preds)
            mae = tf.reduce_mean(delta, axis=0)
            mean_mae = tf.reduce_mean(mae)
            return mean_mae, mae

        @tf.function
        def train_on_batch(dataset_iter):
            inputs, outputs = next(dataset_iter)
            with tf.GradientTape() as tape:
                preds = model(inputs, training=True)
                mean_mae, mae = calculate_mae(outputs, preds)
                loss = mean_mae
            trainer.update_weights(loss, tape)
            return loss, mean_mae, mae

        @tf.function
        def test_on_batch(dataset_iter):
            inputs, outputs = next(dataset_iter)
            preds = model(inputs, training=False)
            mean_mae, mae = calculate_mae(outputs, preds)
            loss = mean_mae
            return loss, mean_mae, mae

        if ex is not None:
            ex.current_run.info = {}
            ex.current_run.info['directory'] = directory
            ex.current_run.info['step'] = []
            ex.current_run.info['mean_mae_train'] = []
            ex.current_run.info['mean_mae_best'] = []
            ex.current_run.info['mean_log_mae_train'] = []
            ex.current_run.info['mean_log_mae_best'] = []

        # Initialize training set error averages
        train['num'] = 0
        train['loss_avg'] = 0.
        train['mae_avg'] = 0.
        train['mean_mae_avg'] = 0.

        def update_average(avg, tmp, num):
            """Incrementally update an average."""
            return avg + (tmp - avg) / num

        # Training loop
        logging.info("Start training")
        steps_per_epoch = int(np.ceil(num_train / batch_size))

        if ckpt_restored is not None:
            step = ckpt.step.numpy()
        else:
            step = 0
        while step <= max_steps:
            # Update step number
            step += 1
            epoch = step // steps_per_epoch
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)

            # Perform training step
            loss, mean_mae, mae = train_on_batch(train['dataset_iter'])

            # Update averages
            train['num'] += 1
            train['loss_avg'] = update_average(
                train['loss_avg'], loss, train['num'])
            train['mae_avg'] = update_average(
                train['mae_avg'], mae, train['num'])
            train['mean_mae_avg'] = update_average(
                train['mean_mae_avg'], mean_mae, train['num'])

            # Save progress
            if (step % save_interval == 0):
                manager.save()

            # Check performance on the validation set
            if (step % validation_interval == 0):
                # Save backup variables and load averaged variables
                trainer.save_variable_backups()
                trainer.load_averaged_variables()

                results = {}
                if num_valid > 0:
                    # Initialize validation set averages
                    validation['num'] = 0
                    validation['loss_avg'] = 0.
                    validation['mae_avg'] = 0.
                    validation['mean_mae_avg'] = 0.

                    # Compute averages
                    for i in range(int(np.ceil(num_valid / batch_size))):

                        loss, mean_mae, mae = test_on_batch(validation['dataset_iter'])

                        validation['num'] += 1
                        validation['loss_avg'] = update_average(
                            validation['loss_avg'], loss, validation['num'])
                        validation['mae_avg'] = update_average(
                            validation['mae_avg'], mae, validation['num'])
                        validation['mean_mae_avg'] = update_average(
                            validation['mean_mae_avg'], mean_mae, validation['num'])

                    # Store results in dictionary
                    results['loss_valid'] = validation['loss_avg']
                    results['mean_mae_valid'] = validation['mean_mae_avg']
                    results['mean_log_mae_valid'] = np.mean(np.log(validation['mae_avg']))
                    for i, key in enumerate(targets):
                        results[key + '_valid'] = validation['mae_avg'][i]

                    if results["mean_mae_valid"] < best_res['mean_mae']:
                        best_res['loss'] = results['loss_valid']
                        best_res['mean_mae'] = results['mean_mae_valid']
                        best_res['mean_log_mae'] = results['mean_log_mae_valid']
                        for i, key in enumerate(targets):
                            best_res[key] = results[key + '_valid']
                        best_res['step'] = step

                        np.savez(best_loss_file, **best_res)
                        model.save_weights(best_ckpt_folder)

                results["loss_best"] = best_res['loss']
                results["mean_log_mae_best"] = best_res['mean_log_mae']
                create_summary(results)

                # Restore backup variables
                trainer.restore_variable_backups()

            # Generate summaries
            if (step % summary_interval == 0) and (step > 0):
                results = {}
                results['loss_train'] = train['loss_avg']
                results['mean_mae_train'] = train['mean_mae_avg']
                results['mean_log_mae_train'] = np.mean(np.log(train['mean_mae_avg']))
                for i, key in enumerate(targets):
                    results[key + '_train'] = train['mae_avg'][i]

                # Reset training set error averages
                train['num'] = 0
                train['loss_avg'] = 0.
                train['mae_avg'] = 0.
                train['mean_mae_avg'] = 0.

                create_summary(results)
                summary_writer.flush()
                ex.current_run.info['step'].append(step)
                ex.current_run.info['mean_mae_train'].append(results['mean_mae_train'])
                ex.current_run.info['mean_mae_best'].append(best_res['mean_mae'])
                ex.current_run.info['mean_log_mae_train'].append(results['mean_log_mae_train'])
                ex.current_run.info['mean_log_mae_best'].append(best_res['mean_log_mae'])

                logging.info(
                    f"{step}/{max_steps} (epoch {epoch+1}): "
                    f"Loss: train={results['loss_train']:.6f}, best={best_res['loss']:.6f}; "
                    f"logMAE: train={results['mean_log_mae_train']:.6f}, best={best_res['mean_log_mae']:.6f}")
    return({"loss": results["loss_train"], "best_loss": best_res['loss'],
            "mean_log_mae": results["mean_log_mae_train"], "best_mean_log_mae": best_res['mean_log_mae'],
            "best_step": best_res['step']})