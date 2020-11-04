from tensorflow.keras.preprocessing.image import ImageDataGenerator
from beartype import beartype


class ImageData:

    @beartype
    def __init__(self, path='../data', validation_split: float = 0.2):
        self.path = path
        self.__data_generator = self.__data_generator(validation_split)

    @staticmethod
    @beartype
    def __data_generator(validation_split: float = 0.2, rescale=1 / 255, zoom_range: float = 0.2,
                         horizontal_flip: bool = True, vertical_flip: bool = True, width_shift_range: float = 0.2,
                         height_shift_range: float = 0.2):
        data_generator = ImageDataGenerator(rescale=rescale,
                                            zoom_range=zoom_range,
                                            horizontal_flip=horizontal_flip,
                                            vertical_flip=vertical_flip,
                                            width_shift_range=width_shift_range,
                                            height_shift_range=height_shift_range,
                                            validation_split=validation_split)
        return data_generator

    def get_train_data(self, batch_size: int = 128, target_size: tuple = (64, 64)):
        train_data = self.__data_generator.flow_from_directory(self.path,
                                                               target_size=target_size,
                                                               batch_size=batch_size,
                                                               class_mode='binary',
                                                               subset='training')
        return train_data

    def get_validation_data(self, batch_size: int = 128, target_size: tuple = (64, 64)):
        validation_data = self.__data_generator.flow_from_directory(self.path,
                                                                    target_size=target_size,
                                                                    batch_size=batch_size,
                                                                    class_mode='binary',
                                                                    subset='validation')
        return validation_data
