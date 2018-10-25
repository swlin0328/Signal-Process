from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Bidirectional, LeakyReLU, concatenate, BatchNormalization, PReLU
from keras.optimizers import RMSprop
import tensorflow as tf

'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=5.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=5.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'


def build_model(input_shape, appliances):
    seq_length = input_shape[0]

    x = Input(shape=input_shape)

    # time_conv
    conv_1 = Conv1D(filters=40, kernel_size=9, strides=1, padding=MODEL_CONV_PADDING)(x)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = PReLU()(conv_1)
    drop_1 = Dropout(0.14)(conv_1)

    conv_2 = Conv1D(filters=40, kernel_size=7, strides=3, padding=MODEL_CONV_PADDING)(drop_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = PReLU()(conv_2)
    drop_2 = Dropout(0.16)(conv_2)

    # freq_conv
    conv_3 = Conv1D(filters=80, kernel_size=5, strides=1, padding=MODEL_CONV_PADDING)(drop_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = PReLU()(conv_3)
    drop_3 = Dropout(0.18)(conv_3)

    conv_4 = Conv1D(filters=80, kernel_size=3, strides=2, padding=MODEL_CONV_PADDING)(drop_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = PReLU()(conv_4)
    drop_4 = Dropout(0.22)(conv_4)

    flat_5 = Flatten()(drop_4)

#===============================================================================================
    # time_conv
    conv_10 = Conv1D(filters=10, kernel_size=31, strides=1, padding=MODEL_CONV_PADDING)(x)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = PReLU()(conv_10)
    drop_10 = Dropout(0.12)(conv_10)

    conv_20 = Conv1D(filters=10, kernel_size=21, strides=10, padding=MODEL_CONV_PADDING)(drop_10)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = PReLU()(conv_20)
    drop_20 = Dropout(0.14)(conv_20)

    # freq_conv
    conv_30 = Conv1D(filters=20, kernel_size=5, strides=1, padding=MODEL_CONV_PADDING)(drop_20)
    conv_30 = BatchNormalization()(conv_30)
    conv_30 = PReLU()(conv_30)
    drop_30 = Dropout(0.16)(conv_30)

    conv_40 = Conv1D(filters=20, kernel_size=3, strides=1, padding=MODEL_CONV_PADDING)(drop_30)
    conv_40 = BatchNormalization()(conv_40)
    conv_40 = PReLU()(conv_40)
    drop_40 = Dropout(0.18)(conv_40)

    flat_50 = Flatten()(drop_40)

#===============================================================================================
    # time_conv
    conv_11 = Conv1D(filters=10, kernel_size=61, strides=1, padding=MODEL_CONV_PADDING)(x)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = PReLU()(conv_11)
    drop_11 = Dropout(0.12)(conv_11)

    conv_21 = Conv1D(filters=10, kernel_size=41, strides=20, padding=MODEL_CONV_PADDING)(drop_11)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = PReLU()(conv_21)
    drop_21 = Dropout(0.12)(conv_21)

    # freq_conv
    conv_31 = Conv1D(filters=20, kernel_size=5, strides=1, padding=MODEL_CONV_PADDING)(drop_21)
    conv_31 = BatchNormalization()(conv_31)
    conv_31 = PReLU()(conv_31)
    drop_31 = Dropout(0.15)(conv_31)

    conv_41 = Conv1D(filters=20, kernel_size=3, strides=1, padding=MODEL_CONV_PADDING)(drop_31)
    conv_41 = BatchNormalization()(conv_41)
    conv_41 = PReLU()(conv_41)
    drop_41 = Dropout(0.15)(conv_41)

    flat_51 = Flatten()(drop_41)

#===============================================================================================
    # time_conv
    conv_12 = Conv1D(filters=20, kernel_size=21, strides=1, padding=MODEL_CONV_PADDING)(x)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = PReLU()(conv_12)
    drop_12 = Dropout(0.14)(conv_12)

    conv_22 = Conv1D(filters=20, kernel_size=13, strides=6, padding=MODEL_CONV_PADDING)(drop_12)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = PReLU()(conv_22)
    drop_22 = Dropout(0.14)(conv_22)

    # freq_conv
    conv_32 = Conv1D(filters=40, kernel_size=5, strides=1, padding=MODEL_CONV_PADDING)(drop_22)
    conv_32 = BatchNormalization()(conv_32)
    conv_32 = PReLU()(conv_32)
    drop_32 = Dropout(0.18)(conv_32)

    conv_42 = Conv1D(filters=40, kernel_size=3, strides=1, padding=MODEL_CONV_PADDING)(drop_32)
    conv_42 = BatchNormalization()(conv_42)
    conv_42 = PReLU()(conv_42)
    drop_42 = Dropout(0.18)(conv_42)

    flat_52 = Flatten()(drop_42)

#===============================================================================================

    conv_0 = Conv1D(filters=4, kernel_size=3, padding='same',activation='linear')(x)
    conv_0 = BatchNormalization()(conv_0)
    drop_0 = Dropout(0.15)(conv_0)
    flat_53 = Flatten()(drop_0)

    # merge
    concate_5 = concatenate([flat_5, flat_50, flat_51, flat_52, flat_53])

    dense_6 = Dense(1720)(concate_5)
    dense_6 = BatchNormalization()(dense_6)
    dense_6 = PReLU()(dense_6)
    drop_6 = Dropout(0.18)(dense_6)

    dense_7 = Dense(1080)(drop_6)
    dense_7 = BatchNormalization()(dense_7)
    dense_7 = PReLU()(dense_7)
    drop_7 = Dropout(0.14)(dense_7)

    reshape_8 = Reshape(target_shape=(seq_length, 9))(drop_7)
    biLSTM_1 = Bidirectional(LSTM(9, dropout=0.1, recurrent_dropout=0.15, return_sequences=True))(reshape_8)
    biLSTM_2 = Bidirectional(LSTM(9, dropout=0.1, recurrent_dropout=0.15, return_sequences=True))(biLSTM_1)

    outputs_disaggregation = []

    for appliance_name in appliances:
        biLSTM_3 = Bidirectional(LSTM(6, dropout=0.1, recurrent_dropout=0.15, return_sequences=True))(biLSTM_2)
    	biLSTM_3 = PReLU()(biLSTM_3)
        outputs_disaggregation.append(TimeDistributed(Dense(1, activation='relu'), name=appliance_name.replace(" ", "_"))(biLSTM_3))

    model = Model(inputs=x, outputs=outputs_disaggregation)
    optimizer = RMSprop(lr=0.001, clipnorm=40)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model
