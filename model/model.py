from keras import Sequential
from keras.layers import BatchNormalization, Conv3D, Activation, LeakyReLU, MaxPooling3D, Dropout, Flatten, Dense
from keras.losses import mean_squared_error
from keras.optimizers import Adam


def create_model(data):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(4, 4, 4), input_shape=(data.shape[1:]), padding='same', name="conv3d_1"))
    model.add(LeakyReLU(name="lrelu_1"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same', name="max_pooling_3d_1"))
    model.add(Dropout(0.2, name="dropout_1"))

    model.add(Conv3D(128, kernel_size=(4, 4, 4), padding='same', name="conv3d_2"))
    model.add(LeakyReLU(name="lrelu_2"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same', name="max_pooling_3d_2"))
    model.add(Dropout(0.2, name="dropout_2"))

    model.add(Conv3D(256, kernel_size=(4, 4, 4), padding='same', name="conv3d_3"))
    model.add(LeakyReLU(name="lrelu_3"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same', name="max_pooling_3d_3"))
    model.add(Dropout(0.2, name="dropout_3"))

    model.add(Flatten(name="flatten_5"))
    model.add(Dense(1024, name="dense_5"))
    model.add(LeakyReLU(name="lrelu_5"))

    model.add(Dense(1024, name="dense_6"))
    model.add(LeakyReLU(name="lrelu_6"))

    model.add(Dense(1, name="dense_7"))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.00005),
                  metrics=['mae'])
    return model
