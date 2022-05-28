import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model import get_unet_model, PSNR
from tensorflow.keras.losses import Huber
import dataset
import json
from history_watcher import HistoryWatcher
from sqrtMSE import SqrtMSE


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args(input_args):
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--image_size", type=int, required=True,
                        help="image size")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--network_depth", type=int, default=4,
                        help="encoder-decoder network depth")
    parser.add_argument("--nb_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--disable_lr_Sched", action='store_true',
                        help="disable learning rate scheduler")
    parser.add_argument("--steps", type=int, default=None,
                        help="steps per epoch")
    parser.add_argument("--val_steps", type=int, default=None,
                        help="steps per validation epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'huber' is expected")
    parser.add_argument("--huber_loss_delta", type=float, default=1.5,
                        help="huber loss delta parameter")
    parser.add_argument("--noise_std", type=float, default=0.015,
                        help="gaussian noise standard deviation")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--infinite_dataset", action='store_true',
                        help="make the dataset infinite")
    args = parser.parse_args(input_args)

    if args.infinite_dataset and ((args.val_steps is None) or (args.steps is None)):
        raise ValueError('"--infinite_dataset" is set but "--val_steps" or "--steps" is None')
    return args


def main(*input_args):
    args = get_args(input_args)
    image_dir = args.image_dir
    test_dir = args.test_dir
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    val_steps = args.val_steps
    loss_type = args.loss
    output_path = Path(args.output_path)
    net_depth = args.network_depth
    noise_std = args.noise_std
    model = get_unet_model(depth=net_depth)
    size = args.image_size
    epoch_to_start = 0

    if args.weight is not None:
        import os.path as path
        model.load_weights(args.weight)
        epoch_to_start = int(path.basename(args.weight).partition('weights.')[2].partition('-')[0])

    opt = Adam(learning_rate=lr)
    callbacks = []

    if loss_type == "huber":
        loss_type = Huber(args.huber_loss_delta)
    if loss_type == "sqrtMSE":
        loss_type = SqrtMSE()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    ##
    train_ds = dataset.create_train_dataset(image_dir, batch_size=batch_size, noise_std=noise_std, image_size=size,inf=args.infinite_dataset)
    val_ds = dataset.create_val_dataset(test_dir, batch_size=batch_size, noise_std=noise_std, image_size=size,inf=args.infinite_dataset)
    ##
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path.joinpath("args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if not args.disable_lr_Sched:
        callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    else:
        print("Learning rate scheduler disabled")
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_weights_only=True,
                                     save_best_only=True))
    callbacks.append(HistoryWatcher(output_path=str(output_path) + "/history_watched.csv"))
    hist = model.fit(
        x=train_ds,
        epochs=nb_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_ds,
        steps_per_epoch=steps,
        validation_steps=val_steps,
        initial_epoch=epoch_to_start
    )

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
