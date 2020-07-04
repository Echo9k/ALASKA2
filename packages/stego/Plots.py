import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img


def plot_minibatch(sample, class_names):
    """
    # DESCRIPTION:
    plot_images will a tuple of (images, classes) from a Keras preprocessing
    tuple of preprocessed images.

    The title of each image will have a number and the type of image:
    Number: corresponding to their index in the batch
    type of image: the correct classification.

    —
    # ARGUMENTS:
    sample_data: A tuple (image, classes)
    The image should be an object of type tensorflow.python.keras.preprocessing.image
    The class an numpy.ndarray

    CLASS_NAMES: The names of the classes.
    """

    def img_type(data, index) -> str:
        for Class in range(len(class_names)):
            if data[1][index][Class] == 1:
                return class_names[i]

    plt.subplots(figsize=(20, 20))
    plt.suptitle('Batch of preprocessed images')
    batch_size = 4
    for i in range(batch_size):
        plt.subplot(4, batch_size // 4, i + 1)
        plt.title(str(i) + ": " + img_type(sample, i))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(array_to_img(sample[0][i]))
