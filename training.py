import os
import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.callbacks import LearningRateScheduler
import glob
import argparse
import time


from models import build_unet


# print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


# device_name = tf.test.gpu_device_name()
# if device_name != "/device:GPU:0":
#     raise SystemError("GPU device not found")
# print("Found GPU at: {}".format(device_name))


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def jaccard_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth
    )


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
    # print(len(image_names))
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
    # val_ds = val_ds.repeat(count=args.count)

    return train_ds, val_ds


def process_tensor(img_path, mask_path):

    raw_im = tf.io.read_file(img_path)
    image = tf.image.decode_png(raw_im, channels=1)
    input_image = tf.cast(image, tf.float32) / 255.0
    input_image = tf.image.resize_with_pad(input_image, 768, 1024, antialias=False)

    raw_ma = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(raw_ma, channels=1)
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


def configure_for_performance(ds, batch_size, augmentation=False, options=True):
    if options:
        options_shard = tf.data.Options()
        options_shard.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
    else:
        options_shard = tf.data.Options()
        options_shard.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )

    ds = ds.map(process_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.with_options(options_shard)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    if augmentation:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def test(model, ds_test):

    # true y from validation dataset
    true_y = np.concatenate([y for x, y in ds_test], axis=0)

    elapsed_eval = time.time()
    y_pred_prob = model.predict(ds_test, args.verbosity)
    elapsed_eval = time.time() - elapsed_eval

    y_pred_thresholded = y_pred_prob > 0.5
    y_pred = y_pred_thresholded.astype(np.int8)
    # print("True, Preds:", set(true_y), set(y_pred))
    iou = jaccard_coef(true_y, y_pred)
    # iou = iou_thresh(true_y, y_pred_thresholded)

    print("elapsed test time, IoU ={:.3f}, {}".format(elapsed_eval, iou))
    sys.stdout.flush()
    return elapsed_eval, iou


def main(args):

    df_save = pd.DataFrame()
    args.distributed = False
    args.world_size = 1
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1

    args.world_rank = args.local_rank = 0

    if args.distributed:
        args.world_rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    args.local_batch_size = math.ceil(args.global_batch_size / args.world_size)

    # only use verbose for master process
    if args.world_rank == 0:
        args.verbosity = 2
    else:
        args.verbosity = 0

    if args.num_intraop_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.num_intraop_threads)
    if args.num_interop_threads:
        tf.config.threading.set_inter_op_parallelism_threads(args.num_interop_threads)

    if args.world_rank == 0:
        print(
            f"Tensorflow get_intra_op_parallelism_threads: {tf.config.threading.get_intra_op_parallelism_threads()}"
        )
        print(
            f"Tensorflow get_inter_op_parallelism_threads: {tf.config.threading.get_inter_op_parallelism_threads()}"
        )
        sys.stdout.flush()

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=args.lr, decay_steps=10000, decay_rate=0.9)

    if args.augment == 0:
        augment = False
    else:
        augment = True

    if args.world_rank == 0:
        print("Tensorflow Settings:")
        settings_map = vars(args)
        for name in sorted(settings_map.keys()):
            print("--" + str(name) + ": " + str(settings_map[name]))
        print("")
        sys.stdout.flush()
    print(args.world_rank, args.local_rank)
    sys.stdout.flush()

    # Device configuration
    args.use_gpu = 1
    l_gpu_devices = [] if args.use_gpu == 0 else tf.config.list_physical_devices("GPU")
    if args.world_rank == 0:
        print("List of GPU devices found:")
        for dev in l_gpu_devices:
            print(str(dev.device_type) + ": " + dev.name)
        print("")
        sys.stdout.flush()

    strategy = None
    if args.world_size == 2 and len(l_gpu_devices) > 0:
        # print("single_node")

        # for single host - multi GPU training MirroredStrategy seems to be much faster than MultiWorkerMirroredStrategy
        strategy = tf.distribute.MirroredStrategy()
    elif args.world_size > 2:
        # print("multi-node")
        if len(l_gpu_devices) > 0:

            # limit to local rank device
            tf.config.set_visible_devices(l_gpu_devices[args.local_rank], "GPU")
            strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=tf.distribute.experimental.CommunicationOptions(
                    implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
                )
            )
        else:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
    if strategy:
        print(strategy, strategy.num_replicas_in_sync)

    # get datasets
    train_ds, val_ds = get_datasets(args, test_size=0.2)

    if args.world_rank == 0:
        print("train iterator size:", len(train_ds), len(val_ds))
        sys.stdout.flush()

    train_ds = configure_for_performance(
        train_ds, args.global_batch_size, augmentation=augment, options=True,
    )
    val_ds = configure_for_performance(
        val_ds, args.global_batch_size, augmentation=augment, options=True,
    )

    if args.world_rank == 0:
        print("train iterator size:", len(train_ds), len(val_ds))
        sys.stdout.flush()

    tf.keras.backend.clear_session()
    if strategy:
        with strategy.scope():
            # model = get_model(args, input_shape)
            # cur_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(cur_optimizer)
            model = build_unet((768, 1024, 1), 1)
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=Adam(learning_rate=0.001),  # lr_schedule)
                metrics=["accuracy"],
            )

    else:
        # cur_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(cur_optimizer)
        model = build_unet((768, 1024, 1), 1)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=Adam(learning_rate=0.001),  # lr_schedule)
            metrics=["accuracy"],
        )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, mode="min"
    )
    K = args.world_size
    init_lr = 0.001
    increment = ((init_lr * K) - init_lr) / 10

    # designed for 150 epochs
    def lr_schedule(epoch, lr):

        if epoch < 80:
            return init_lr * K
        # if epoch < 10:
        #     return epoch * increment + init_lr
        # elif epoch >= 10 and epoch < 40:
        #     return init_lr * K
        elif epoch >= 80 and epoch < 120:
            return init_lr / 2 * K
        elif epoch >= 120 and epoch <= 150:
            return init_lr / 4 * K
        # elif epoch >= 45:
        #     return init_lr / 8 * K

    scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # csv_logger = CSVLogger("log.csv", append=True, separator=";")
    time_callback = TimeHistory()
    start_time = time.time()
    history = model.fit(
        train_ds,
        verbose=args.verbosity,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[time_callback, scheduler],
        #  CustomLearningRateScheduler(lr_schedule)],
    )
    end_time = time.time() - start_time

    if args.world_rank == 0:

        # model.save(str(os.environ["WORK"]) + "/models_tf/" + str(args.world_size))

        df_save["time_per_epoch"] = time_callback.times
        df_save["loss"] = history.history["loss"]
        df_save["val_loss"] = history.history["val_loss"]
        df_save["lr"] = history.history["lr"]
        df_save["training_time"] = end_time
        print("Elapsed execution time: " + str(end_time) + " sec")
        test_time, iou = test(model, val_ds)
        df_save["test_time"] = test_time
        df_save["iou"] = iou
        sys.stdout.flush()

    df_save.to_csv("./log.csv", sep=",", float_format="%.6f")

    print(args.world_rank)
    print("Elapsed execution time: " + str(end_time) + " sec")
    test(model, val_ds)
    sys.stdout.flush()

    # if args.world_rank == 0:
    #     print("here")
    #     print("Elapsed execution time: " + str(end_time) + " sec")
    #     test(model, val_ds)
    #     sys.stdout.flush()

    # if strategy:
    #     import atexit

    #     #  # atexit.register(strategy._extended._collective_ops._pool.close)
    #     atexit.register(strategy._extended._host_cross_device_ops._pool.close)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training args")
    parser.add_argument("--global_batch_size", type=int, help="8 or 16 or 32")
    # parser.add_argument("--device", type=str, help="cuda" or "cpu", default="cuda")
    parser.add_argument("--lr", type=float, help="ex. 0.001", default=0.001)
    parser.add_argument("--count", type=int, help="for dataset repeat", default=2)
    parser.add_argument("--epochs", type=int, help="iterations")

    parser.add_argument(
        "--image_dir",
        type=str,
        help="directory of images",
        default=str(os.environ["WORK"]) + "/images_collective",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="directory of masks",
        default=str(os.environ["WORK"]) + "/masks_collective",
    )

    parser.add_argument("--augment", type=int, help="0 is False, 1 is True", default=0)

    parser.add_argument(
        "--num_interop_threads",
        required=False,
        help="Number of interop threads",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_intraop_threads",
        required=False,
        help="Number of intra-op threads",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    main(args)
