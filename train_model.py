import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)

from IPython import display

from utils import (
    squeeze,
    get_spectrogram,
    plot_spectrogram,
    make_spec_ds,
    RNN,
    step_decay,
)

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = "./data/"

# Train-validation split
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

# Test-validation split
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# Creating spectrogram datasets
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

input_shape = example_spectrograms.shape[1:]
print("Input shape:", input_shape)
num_labels = len(label_names)

# Training model
model = RNN(input_shape, num_labels, train_spectrogram_ds)

lrate = LearningRateScheduler(step_decay)
earlystopper = EarlyStopping(
    monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True
)
checkpointer = ModelCheckpoint(
    "model_best_3.h5", monitor="val_accuracy", verbose=1, save_best_only=True
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

metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
plt.legend(["loss", "val_loss"])
plt.ylim([0, max(plt.ylim())])
plt.xlabel("Epoch")
plt.ylabel("Loss [CrossEntropy]")

plt.subplot(1, 2, 2)
plt.plot(
    history.epoch,
    100 * np.array(metrics["accuracy"]),
    100 * np.array(metrics["val_accuracy"]),
)
plt.legend(["accuracy", "val_accuracy"])
plt.ylim([0, 100])
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")

plt.savefig("loss_acc_3.png")

model.save("model_last_3.h5")
