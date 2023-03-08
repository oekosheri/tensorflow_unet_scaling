import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import argparse
import time
from models import build_unet
import horovod
import horovod.tensorflow.keras as hvd
from tensorflow.keras import mixed_precision


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def jaccard_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    #y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = y_pred.flatten()
    #y_pred_f = tf.keras.layers.Flatten()(y_pred)
    #y_true_f = tf.cast(y_true_f, tf.float16)
    #y_pred_f = tf.cast(y_pred_f, tf.float16)
    #intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    #intersection = tf.cast(intersection, tf.float16)
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth
    )
    #iou = (intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + smooth)
    #print(iou)
    #y_true_f_sum = tf.keras.backend.sum(y_true_f)
    #y_true_f_sum = tf.cast(y_true_f_sum, tf.float16)
    #y_pred_f_sum = tf.keras.backend.sum(y_pred_f)
    #y_pred_f_sum = tf.cast(y_pred_f_sum, tf.float16)
    #return (intersection + smooth) / (y_true_f_sum + y_pred_f_sum - intersection + smooth)


def jaccard_coef_loss(y_true, y_pred, smooth=1.0):
    return -jaccard_coef(y_true, y_pred, smooth=1.0)


#https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
#https://www.tensorflow.org/api_docs/python/tf/keras/metrics/IoU
"""def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth"""


def iou_thresh(y_true, y_pred_thresholded):
    y_true_f = y_true.flatten()
    y_pred_thresholded_f = y_pred_thresholded.flatten()
    intersect = np.logical_and(y_true_f, y_pred_thresholded_f)
    union = np.logical_or(y_true_f, y_pred_thresholded_f)
    iou_score = np.sum(intersect) / np.sum(union)
    return iou_score


def get_datasets(args, test_size=0.2):

    image_names = glob.glob(args.image_dir + "/*.png")
    image_names.sort()
    image_subset = image_names[0:]
    #print(len(image_names))
    sys.stdout.flush()

    mask_names = glob.glob(args.mask_dir + "/*.png")
    mask_names.sort()
    mask_subset = mask_names[0:]

    assert len(image_subset) == len(mask_subset)

    # make tensorflow dataset
    ds = tf.data.Dataset.from_tensor_slices((image_subset, mask_subset))
    ds = ds.shuffle(len(image_subset), reshuffle_each_iteration=True)
    # ds = ds.repeat(count=args.count)

    val_size = int(len(image_subset) * test_size)
    train_ds = ds.skip(val_size)
    val_ds = ds.take(val_size)

    # repeat
    train_ds = train_ds.repeat(count=args.count)
    #val_ds = val_ds.repeat(count=args.count)

    return train_ds, val_ds


def process_tensor(img_path, mask_path):

    raw_im = tf.io.read_file(img_path)
    image = tf.image.decode_png(raw_im, channels=1)
    input_image = tf.cast(image, tf.float32) / 255.0
    input_image = tf.image.resize_with_pad(input_image, 768, 1024, antialias=False)

    raw_ma = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(raw_ma, channels=1)
    input_mask = tf.cast(mask, tf.float32) / 255.0
    input_mask = tf.image.resize_with_pad(input_mask, 768, 1024, antialias=False)
    # input_mask = tf.cast(input_mask > 0.2, tf.int8)

    return input_image, input_mask


def augment(image, mask):

    # deterministic flip
    # image = tf.image.stateless_random_flip_left_right(image, seed=(1, 2))
    # mask = tf.image.stateless_random_flip_left_right(mask, seed=(1, 2))

    image = tf.image.random_brightness(image, max_delta=40.0 / 255.0)
    image = tf.image.random_contrast(image, 0.2, 1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    mask = tf.clip_by_value(mask, 0, 1)

    return image, mask


def configure_for_performance(ds, batch_size, augmentation=False, shard=False):
    
    if shard:
        ds = ds.shard(hvd.size(), hvd.rank())

    ds = ds.map(process_tensor, num_parallel_calls=tf.data.AUTOTUNE)

    #ds = ds.repeat(args.epochs)

    # calculate buffer size for shuffle operation
    #bsize = int((9000/hvd.size())+1000)
    #ds = ds.shuffle(buffer_size=bsize, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)

    #ds = ds.with_options(options_shard)

    if augmentation:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


class IOUTestCallback(tf.keras.callbacks.Callback):

    def __init__(self, ds_test):
        self.ds_test = ds_test
    
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        _, iou = test(self.model, self.ds_test)
        tf.summary.scalar('IoU', data=iou, step=epoch)


def test(model, ds_test):

    # true y from validation dataset
    true_y = np.concatenate([y for x, y in ds_test], axis=0)

    elapsed_eval = time.time()
    if hvd.rank() == 0:
        verbose = 1
    else:
        verbose = 0
    y_pred_prob = model.predict(ds_test, verbose)
    elapsed_eval = time.time() - elapsed_eval

    y_pred_thresholded = y_pred_prob > 0.5
    y_pred = y_pred_thresholded.astype(np.int8)
    # print("True, Preds:", set(true_y), set(y_pred))
    iou = jaccard_coef(true_y, y_pred)
    # iou = iou_thresh(true_y, y_pred_thresholded)

    #print("elapsed test time, IoU ={:.3f}, {}".format(elapsed_eval, iou))
    #sys.stdout.flush()
    return elapsed_eval, iou


def main(args):

    # set environment variable, prevent tf.data from interfering the threads that launch kernels on the GPU
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

    # enable mixed precision
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    # tensor float32 hardware support
    tf.config.experimental.enable_tensor_float_32_execution(True)

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    df_save = pd.DataFrame()
    
    local_batch_size = args.local_batch_size
    # for strong scaling
    #https://developer.hpe.com/blog/scaling-deep-learning-workloads/
    #local_batch_size = int(global_batch_size / hvd.size())
    lr = args.lr

    # DEBUG CODE, only print on worker 0
    if hvd.rank() == 0:
        print("Tensorflow version:", tf.__version__)
        print("Number of GPUs available:", len(tf.config.experimental.list_physical_devices("GPU")))
        print("HVD RANKS:",hvd.size())
        print("LOCAL BATCH SIZE:",local_batch_size)
        print("GLOBAL BATCH",local_batch_size*hvd.size())
        print("LR",lr)

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=args.lr, decay_steps=10000, decay_rate=0.9)

    # get datasets
    train_ds, val_ds = get_datasets(args, test_size=0.2)

    #if hvd.rank() == 0:
        #print("train iterator size:", len(train_ds), len(val_ds))
        #sys.stdout.flush()

    train_ds = configure_for_performance(
        train_ds, local_batch_size, augmentation=augment, shard=False,
    )

    val_ds = configure_for_performance(
        val_ds, local_batch_size, augmentation=False, shard=False,
    )

    model = build_unet((768, 1024, 1), 1)

    scaled_lr = lr * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    # In tensorflow 2.11 there is an IoU metric build in ... maybe use that one instead
    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/IoU
    # In my opinion the loss function and the metric should be custom, i.e., the jaccard_coef function...
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    #if hvd.rank() == 0:
    #    callbacks.append(tf.keras.callbacks.ModelCheckpoint('checkpoints/checkpoint-{epoch}.h5'))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # add early stopping callback to save computing resources 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    if args.no_early_stop == True:
        callbacks.append(early_stopping)

    # tensorboard callback to track loss
    logdir = "logs/"
    logname = args.log_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir+str(logname))
    if hvd.rank() == 0:
        callbacks.append(tensorboard_callback)

    # custom scalar log for iou
    file_writer = tf.summary.create_file_writer(logdir+"/metrics/"+str(logname))
    file_writer.set_as_default()
    if hvd.rank() == 0:
        callbacks.append(IOUTestCallback(val_ds))

    # create learning rate sheduler
    #increment = ((scaled_lr) - scaled_lr) / 10
    def scheduler(epoch, lr):
        if epoch < 80:
            return scaled_lr
        # if epoch < 10:
        #     return epoch * increment + scaled_lr
        # elif epoch >= 10 and epoch < 40:
        #     return scaled_lr
        elif epoch >= 80 and epoch < 120:
            return scaled_lr / 2
        elif epoch >= 120 and epoch <= 150:
            return scaled_lr / 4
        # elif epoch >= 45:
        #     return scaled_lr / 8

    lr_sheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks.append(lr_sheduler_callback)
    
    #time_callback = TimeHistory()
    #if hvd.rank() == 0:
    #    callbacks.append(time_callback)
    
    start_time = time.time()

    history = model.fit(
        train_ds,
        verbose=verbose,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    end_time = time.time() - start_time

    if hvd.rank() == 0:

        df_save["loss"] = history.history["loss"]
        df_save["accuracy"] = history.history["accuracy"]
        df_save["val_loss"] = history.history["val_loss"]
        df_save["val_accuracy"] = history.history["val_accuracy"]
        df_save["lr"] = history.history["lr"]
        df_save["training_time"] = end_time*hvd.size()
        df_save["ranks"] = hvd.size()
        df_save.to_csv("timer/"+str(logname)+".csv", sep=",", float_format="%.6f")

    print("Elapsed execution time per rank: " + str(end_time) + " sec")
    print("Elapsed total execution time: " + str(end_time*hvd.size()) + " sec")

    # save the trained model
    pos = logname.find("_")
    model_name = logname[pos+1:]
    model_name = "model_"+model_name
    if args.save_model:
        if hvd.rank() == 0:
            model.save("models/"+str(model_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training args")
    parser.add_argument("--local_batch_size", type=int, help="Batch size per rank for weak scaling, default 64", default=64)
    parser.add_argument("--lr", type=float, default=0.001, help="ex. 0.001")
    parser.add_argument("--count", type=int, help="for dataset repeat", default=2)
    parser.add_argument("--epochs", type=int, default=150, help="iterations")
    parser.add_argument("--log_name", type=str, help="name for the log file")
    parser.add_argument("--no_early_stop", type=bool, help="enable/disable early stopping", default=False)
    parser.add_argument("--save_model", type=bool, help="If set save the model", default=False)

    parser.add_argument(
        "--image_dir",
        type=str,
        help="directory of images",
        required=True
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="directory of masks",
        required=True
    )

    parser.add_argument(
        "--augment", 
        type=bool,
        help="If set use data augmentation",
        default=False
    )

    args = parser.parse_args()

    main(args)
