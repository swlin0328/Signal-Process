from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import RMSprop


MODEL_CONV_FILTERS = 32
MODEL_CONV_KERNEL_SIZE = 18
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'


def build_model(input_shape):
    seq_length = input_shape[0]

    # build it!
    model = Sequential()

    # conv
    model.add(Conv1D(input_shape=input_shape, filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=64, kernel_size=12, padding=MODEL_CONV_PADDING))
    model.add(Activation('relu'))
    model.add(Dropout(0.14))

    model.add(Conv1D(filters=128, kernel_size=8, padding=MODEL_CONV_PADDING))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.18))

    model.add(Conv1D(filters=128, kernel_size=5, padding=MODEL_CONV_PADDING))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # reshape
    model.add(Flatten())

    model.add(Dense(1280))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(960))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(720))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Reshape(target_shape=(seq_length, 12)))

    model.add(Bidirectional(LSTM(6, return_sequences=True)))
    model.add(Bidirectional(LSTM(3, return_sequences=True)))

    model.add(TimeDistributed(Dense(1)))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    optimizer = RMSprop(lr=0.001, clipnorm=10)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc', 'mae', 'mse'])

    return model
