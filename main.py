import os
import pathlib

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import tensorflow as tf

from IPython import display

from utils import squeeze, get_spectrogram, plot_spectrogram, make_spec_ds, RNN

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = "./data/"

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset="both",
)

label_names = np.array(train_ds.class_names)
print("label names:", label_names)

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

input_shape = example_spectrograms.shape[1:]
print("Input shape:", input_shape)
num_labels = len(label_names)

model = RNN(input_shape, num_labels, train_spectrogram_ds)

# EPOCHS = 10
# history = model.fit(
#     train_spectrogram_ds,
#     validation_data=val_spectrogram_ds,
#     epochs=EPOCHS,
#     callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
# )

import math
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    if lrate < 4e-5:
        lrate = 4e-5

    print("Changing learning rate to {}".format(lrate))
    return lrate


lrate = LearningRateScheduler(step_decay)

earlystopper = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    verbose=1,
    restore_best_weights=True,
)
checkpointer = ModelCheckpoint(
    "model_best.h5",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
)

history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=60,
    use_multiprocessing=False,
    workers=4,
    verbose=1,
    callbacks=[earlystopper, checkpointer, lrate],
)

model.save("model_last.h5")
