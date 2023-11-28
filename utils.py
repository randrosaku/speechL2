import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, LSTM, Dense, Normalization

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def plot_spectrogram(spectrogram, ax):
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
    return ds.map(
        map_func=lambda audio,label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
    
def RNN(input_shape, num_labels, data):
    
    norm_layer = Normalization()
    norm_layer.adapt(data.map(map_func=lambda spec, label: spec))

    model = Sequential([
        Input(shape=input_shape),
        norm_layer,
        Lambda(lambda q: tf.squeeze(q, -1), name='squeeze_last_dim'),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(32),
        Dense(num_labels, activation='softmax'),
    ])
    
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])

    model.summary()
    
    return model