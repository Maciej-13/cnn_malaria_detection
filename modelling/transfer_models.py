from tensorflow import keras
from tensorflow.keras import layers
from beartype import beartype

from modelling.cnn_models import CNN


class PreTrainedModel:

    def __init__(self):
        self._base_model = self.__create_base_model()

    def number_of_layers_in_base_model(self):
        return len(self._base_model.layers)

    def unfreeze_top_layers(self, unfreeze_from: int):
        for layer in self._base_model.layers[unfreeze_from:]:
            layer.trainable = True

    def get_base_model_summary(self):
        return self._base_model.summary()

    def fine_tuning(self, training_data, initial_epochs: int, fine_tune_epochs: int, **kwargs):
        raise NotImplementedError("Method must be overridden in the derived class!")

    @beartype
    def __create_base_model(self, **kwargs):
        base_model = keras.Sequential([])
        return base_model


class VGG19(CNN, PreTrainedModel):

    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._base_model = self.__create_base_model(input_shape)
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    def fine_tuning(self, training_data, initial_epochs: int, fine_tune_epochs: int, **kwargs):
        epochs = fine_tune_epochs + initial_epochs
        return self._model.fit(training_data, epochs=epochs, initial_epoch=initial_epochs, **kwargs)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        inputs = keras.Input(shape=input_shape)
        x = self._base_model(inputs, training=False)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Flatten()(x)
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        return model

    @beartype
    def __create_base_model(self, input_shape: tuple):
        base_model = keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
        base_model.trainable = False
        return base_model


class ResNet50(CNN, PreTrainedModel):

    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._base_model = self.__create_base_model(input_shape)
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    def fine_tuning(self, training_data, initial_epochs: int, fine_tune_epochs: int, **kwargs):
        epochs = fine_tune_epochs + initial_epochs
        return self._model.fit(training_data, epochs=epochs, initial_epoch=initial_epochs, **kwargs)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        inputs = keras.Input(shape=input_shape)
        x = self._base_model(inputs, training=False)
        x = layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        return model

    @beartype
    def __create_base_model(self, input_shape: tuple):
        base_model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        base_model.trainable = False
        return base_model


class InceptionResNetV2(CNN, PreTrainedModel):

    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._base_model = self.__create_base_model(input_shape)
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    def fine_tuning(self, training_data, initial_epochs: int, fine_tune_epochs: int, **kwargs):
        epochs = fine_tune_epochs + initial_epochs
        return self._model.fit(training_data, epochs=epochs, initial_epoch=initial_epochs, **kwargs)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        inputs = keras.Input(shape=input_shape)
        x = self._base_model(inputs, training=False)
        x = keras.layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        return model

    @beartype
    def __create_base_model(self, input_shape: tuple):
        base_model = keras.applications.InceptionResNetV2(input_shape=input_shape, include_top=False,
                                                          weights='imagenet')
        base_model.trainable = False
        return base_model


class NasNetMobile(CNN, PreTrainedModel):

    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._base_model = self.__create_base_model(input_shape)
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    def fine_tuning(self, training_data, initial_epochs: int, fine_tune_epochs: int, **kwargs):
        epochs = fine_tune_epochs + initial_epochs
        return self._model.fit(training_data, epochs=epochs, initial_epoch=initial_epochs, **kwargs)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        inputs = keras.Input(shape=input_shape)
        x = self._base_model(inputs, training=False)
        x = keras.layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        return model

    @beartype
    def __create_base_model(self, input_shape: tuple):
        base_model = keras.applications.NASNetMobile(include_top=False, input_shape=input_shape, weights='imagenet')
        base_model.trainable = False
        return base_model


class MobileNetV2(CNN, PreTrainedModel):

    def __init__(self, input_shape: tuple, activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        self._base_model = self.__create_base_model(input_shape)
        self._model = self.__create_model(input_shape, activation, dropout_rate)

    def fine_tuning(self, training_data, initial_epochs: int, fine_tune_epochs: int, **kwargs):
        epochs = fine_tune_epochs + initial_epochs
        return self._model.fit(training_data, epochs=epochs, initial_epoch=initial_epochs, **kwargs)

    @beartype
    def __create_model(self, input_shape: tuple, activation: str, dropout_rate: float):
        inputs = keras.Input(shape=input_shape)
        x = self._base_model(inputs, training=False)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        return model

    @beartype
    def __create_base_model(self, input_shape: tuple):
        base_model = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False
        return base_model
