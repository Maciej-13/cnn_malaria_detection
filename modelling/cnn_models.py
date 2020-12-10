from tensorflow import keras
from tensorflow.keras import layers
from beartype import beartype


class CNN:

    def __init__(self):
        self._model = self.__create_model()

    def get_summary(self):
        return self._model.summary()

    @beartype
    def compile_model(self, loss, optimizer, metrics: list):
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @beartype
    def train(self, training_data, epochs: int, steps_per_epoch: int, **kwargs):
        return self._model.fit(training_data, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)

    def evaluate(self, data_generator):
        return self._model.evaluate(data_generator)

    def predict(self, data):
        return self._model.predict(data)

    def __create_model(self, **kwargs):
        model = keras.Sequential([])
        return model


class TwoLayers(CNN):

    @beartype
    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        model = keras.Sequential([
            layers.Conv2D(input_shape=input_shape, filters=16, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Flatten(),

            layers.Dense(units=128),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(1, activation='sigmoid')
        ])
        return model


class ThreeLayers(CNN):

    @beartype
    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        model = keras.Sequential([
            layers.Conv2D(input_shape=input_shape, filters=16, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Flatten(),

            layers.Dense(units=128),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(1, activation='sigmoid')
        ])
        return model


class LeNet5(CNN):

    @beartype
    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        model = keras.Sequential([
            layers.Conv2D(filters=6, kernel_size=(3, 3), input_shape=input_shape, padding="same"),
            layers.Activation(activation),
            layers.AveragePooling2D(),

            layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),
            layers.AveragePooling2D(),

            layers.Flatten(),

            layers.Dense(units=248),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=124),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=1),
            layers.Activation('sigmoid')
        ])
        return model


class AlexNet(CNN):

    @beartype
    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        model = keras.Sequential([
            layers.Conv2D(filters=4, kernel_size=(3, 3), input_shape=input_shape, padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"),

            layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activation, padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            layers.Flatten(),

            layers.Dense(units=256),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=256),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=128),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=1),
            layers.Activation('sigmoid'),
        ])
        return model


class VGG16(CNN):

    @beartype
    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        model = keras.Sequential([
            layers.Conv2D(input_shape=input_shape, filters=4, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same"),
            layers.Activation(activation),

            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Flatten(),

            layers.Dense(units=256),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=128),
            layers.Activation(activation),
            layers.Dropout(rate=dropout_rate),

            layers.Dense(units=1),
            layers.Activation("sigmoid"),

        ])
        return model
