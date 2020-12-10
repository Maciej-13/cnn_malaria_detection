from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import random
from beartype import beartype


class ImageDataReader:

    @beartype
    def __init__(self, path: str = '/data', validation_split: float = 0.2, random_seed: int = 12345,
                 batch_size: int = 32, target_size: tuple = (80, 80), **kwargs):
        self.__path = path
        self.__seed = random_seed
        self.__set_seed()
        self.__batch = batch_size
        self.__target = target_size
        self.__data_generator = self.__data_generator(validation_split, **kwargs)

    @beartype
    def get_train_data(self):
        train_data = self.__data_generator.flow_from_directory(self.__path,
                                                               seed=self.__seed,
                                                               target_size=self.__target,
                                                               batch_size=self.__batch,
                                                               class_mode='binary',
                                                               subset='training')
        return train_data

    @beartype
    def get_validation_data(self):
        validation_data = self.__data_generator.flow_from_directory(self.__path,
                                                                    seed=self.__seed,
                                                                    target_size=self.__target,
                                                                    batch_size=self.__batch,
                                                                    class_mode='binary',
                                                                    subset='validation')
        return validation_data

    @beartype
    def __set_seed(self):
        random.set_seed(self.__seed)

    @staticmethod
    @beartype
    def __data_generator(validation_split: float, **kwargs):
        data_generator = ImageDataGenerator(validation_split=validation_split, **kwargs)
        return data_generator
