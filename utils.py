import numpy as np
import matplotlib as plt
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, LSTM, Dense, Normalization


def squeeze(audio, labels):
    """
    Drop the extra axis representing audio channels
    """
    audio = tf.squeeze(audio, axis=-1)

    return audio, labels


def get_spectrogram(waveform):
    """
    Convert the waveform to a spectrogram via a STFT
    """
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram


def plot_spectrogram(spectrogram, ax):
    """
    Function to plot spectrogram
    """
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)

    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def make_spec_ds(ds):
    """
    Create spectrogram datasets from the audio datasets
    """
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def RNN(input_shape, num_labels, data):
    """
    Recurrent Neural Network (RNN) with Long Short-Term Memory layers

    Args:
        input_shape: tensor shape to the model
        num_labels: number of categorical classes
        data: training spectrogram dataset

    Returns:
        model: RNN tensorflow model with LSTM layers
    """
    norm_layer = Normalization()
    norm_layer.adapt(data.map(map_func=lambda spec, label: spec))

    model = Sequential(
        [
            Input(shape=input_shape, name="Input layer"),
            norm_layer,
            Lambda(lambda q: tf.squeeze(q, -1), name="squeeze_last_dim"),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(64, activation="relu"),
            Dense(32),
            Dense(num_labels, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss=["sparse_categorical_crossentropy"], metrics=["accuracy"]
    )

    model.summary()

    return model


def step_decay(epoch):
    """
    Learning rate scheduler
    """
    initial_lrate = 0.0005
    drop = 0.3
    epochs_drop = 20
    new_lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    if new_lrate < 4e-5:
        new_lrate = 4e-5

    print(f"Changing learning rate to {new_lrate}")

    return new_lrate


def update_counts(ds, counts, label_names):
    """
    Counting number of class elements for each dataset
    """
    for images, labels in ds:
        for label in labels.numpy():
            label_name = label_names[label]
            counts[label_name] += 1


def plot_curves(history):
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

    plt.savefig("loss_acc_2.png")
