from beartype import beartype
import matplotlib.pyplot as plt


class ImageDataShow:

    @staticmethod
    @beartype
    def show_image(image, label, infected_class: int = 1):
        plt.imshow(image)
        plt.title('Infected' if label == infected_class else 'Not infected')
        plt.show()

    @staticmethod
    @beartype
    def show_batch(images, labels, infected_class: int = 1):
        text_labels = ['Infected' if x == infected_class else 'Not infected' for x in labels]
        plt.figure(figsize=(18, 8))

        for i in range(32):
            plt.subplot(4, 8, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])
            plt.xlabel(text_labels[i])
        plt.show()


class VisualizePerformance:

    @staticmethod
    @beartype
    def plot_training(epochs: int, accuracy: list, loss: list):
        epochs_range = range(epochs)
        fig, (ax1, ax2) = plt.subplots(2)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        ax1.plot(epochs_range, accuracy[0], label='Training Accuracy')
        ax1.plot(epochs_range, accuracy[1], label='Validation Accuracy')
        ax1.legend(loc='lower right')
        ax1.set_title('Training and Validation Accuracy')
        ax2.plot(epochs_range, loss[0], label='Training Loss')
        ax2.plot(epochs_range, loss[1], label='Validation Loss')
        ax2.legend(loc='upper right')
        ax2.set_title('Training and Validation Loss')
        plt.show()

    @staticmethod
    @beartype
    def plot_fine_tuning(initial_epochs: int, tuning_epochs: int, accuracy: list, loss: list):
        epochs_range = range(initial_epochs + tuning_epochs)
        fig, (ax1, ax2) = plt.subplots(2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        ax1.plot(epochs_range, accuracy[0], label='Training Accuracy')
        ax1.plot(epochs_range, accuracy[1], label='Validation Accuracy')
        ax1.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
        ax1.legend(loc='lower right')
        ax1.set_title('Training and Validation Accuracy')

        ax2.plot(epochs_range, loss[0], label='Training Loss')
        ax2.plot(epochs_range, loss[1], label='Validation Loss')
        ax2.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
        ax2.legend(loc='upper right')
        ax2.set_title('Training and Validation Loss')

        plt.show()

