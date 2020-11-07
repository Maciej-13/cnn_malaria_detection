from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from beartype import beartype
import matplotlib.pyplot as plt


class ImageDataHandler:

    @beartype
    def __init__(self, path='/data', validation_split: float = 0.2, random_seed: int = 12345):
        self.__path = path
        self.__seed = random_seed
        self.__set_seed()
        self.__data_generator = self.__data_generator(validation_split)

    @beartype
    def get_train_data(self, batch_size: int = 32, target_size: tuple = (80, 80)):
        train_data = self.__data_generator.flow_from_directory(self.__path,
                                                               seed=self.__seed,
                                                               target_size=target_size,
                                                               batch_size=batch_size,
                                                               class_mode='binary',
                                                               subset='training')
        return train_data

    @beartype
    def get_test_data(self, batch_size: int = 32, target_size: tuple = (80, 80)):
        test_data = self.__data_generator.flow_from_directory(self.__path,
                                                              seed=self.__seed,
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              class_mode='binary',
                                                              subset='validation')
        return test_data

    @beartype
    def __set_seed(self):
        tf.random.set_seed(self.__seed)

    @staticmethod
    @beartype
    def __data_generator(validation_split: float,
                         rescale=1/255,
                         zoom_range: float = 0.0,
                         horizontal_flip: bool = False,
                         vertical_flip: bool = False,
                         width_shift_range: float = 0.0,
                         height_shift_range: float = 0.0):
        data_generator = ImageDataGenerator(rescale=rescale,
                                            zoom_range=zoom_range,
                                            horizontal_flip=horizontal_flip,
                                            vertical_flip=vertical_flip,
                                            width_shift_range=width_shift_range,
                                            height_shift_range=height_shift_range,
                                            validation_split=validation_split)
        return data_generator

    @staticmethod
    def show_image(image, label, infected_class: int = 1):
        plt.imshow(image)
        plt.title('Infected' if label == infected_class else 'Not infected')
        plt.show()

    @staticmethod
    def show_batch(images, labels, batch_size: int = 32, infected_class: int = 1):
        text_labels = ['Infected' if x == infected_class else 'Not infected' for x in labels]
        for i in range(0, batch_size, 4):
            f, ax = plt.subplots(2, 4)
            ax[0, 0].get_xaxis().set_visible(False)
            ax[0, 0].imshow(images[i])
            ax[0, 0].set_title(text_labels[i])

            ax[0, 1].get_xaxis().set_visible(False)
            ax[0, 1].imshow(images[i + 1])
            ax[0, 1].set_title(text_labels[i + 1])

            ax[1, 0].get_xaxis().set_visible(False)
            ax[1, 0].imshow(images[i + 2])
            ax[1, 0].set_title(text_labels[i + 2])

            ax[1, 1].get_xaxis().set_visible(False)
            ax[1, 1].imshow(images[i + 3])
            ax[1, 1].set_title(text_labels[i + 3])
