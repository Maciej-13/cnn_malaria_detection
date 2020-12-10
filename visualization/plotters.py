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
    def show_batch(images, labels, batch_size: int = 32, infected_class: int = 1):
        text_labels = ['Infected' if x == infected_class else 'Not infected' for x in labels]
        for i in range(0, batch_size, 4):
            f, ax = plt.subplots(2, 2)
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
        plt.show()
