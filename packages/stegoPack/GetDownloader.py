from numpy.random import choice
from os import walk
from typing import Dict, Tuple, Optional, List, Generator
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import image, keras, split, truediv, subtract, reduce_min, reduce_max


def yuv_img_compare(batch: Generator, sample_size: int = 5):

    yuv_images, _ = next(batch)
    yuv_images = yuv_images[:sample_size]
    last_dimension_axis = 3
    yuv_tensor_images = truediv(
        subtract(
            yuv_images,
            reduce_min(yuv_images)
        ),
        subtract(
            reduce_max(yuv_images),
            reduce_min(yuv_images)
        )
    )
    y, u, v = split(yuv_tensor_images, 3, axis=last_dimension_axis)
    target_uv_min, target_uv_max = -0.5, 0.5
    u = u * (target_uv_max - target_uv_min) + target_uv_min
    v = v * (target_uv_max - target_uv_min) + target_uv_min
    return y, u, v


class GetData:
    def __init__(self, dir_url: Dict[str, str] = None, class_names: [List] = None, input_directory: str = '/content',
                 subdirectory: str = '/stego_images'):
        self.dir_url = dir_url
        self.class_names = class_names
        self.img_directory = input_directory + subdirectory

    @staticmethod
    def unique_files(dir_url, img_directory, class_names) -> Dict:
        _, dirnames, _ = next(walk(img_directory))

        if class_names is None:
            class_names = set(dir_url)
        if all([class_i in dirnames for class_i in class_names]):
            print("All directories exist.")
            while True:
                user_check = input('Force download y/n: ').lower()
                if 'n' == user_check:
                    break
                elif 'y' == user_check:
                    return dir_url
                else:
                    'Answer: y/n'

        duplicate_folders = set(dir_url).intersection(dirnames)
        print(f'Duplicate : {duplicate_folders}')
        [dir_url.pop(i) for i in duplicate_folders]
        return dir_url

    def deduce_class_names(self):
        img_directory = self.img_directory
        if self.class_names is None:
            _, directories, _ = next(walk(img_directory))
            self.class_names = (None, directories)[len(directories) > 0]

    def download_unzip(self, get_all=True) -> None or Generator:
        """
        Downloads data from the dir_url of the form {category, url}.
        Stores each folder under input_directory/subdirectory/category/
        """

        def mk_params(dir_key, file_url):
            params = {'fname': dir_key, 'origin': file_url,
                      'cache_subdir': self.img_directory + dir_key,
                      'hash_algorithm': 'auto', 'extract': True,
                      'archive_format': 'auto', 'cache_dir': None}
            return params

        to_download = self.unique_files(self.dir_url, self.img_directory, self.class_names)

        if get_all:
            for key, url in to_download.items():
                f_params = mk_params(key, url)
                keras.utils.get_file(**f_params)
        else:
            return (keras.utils.get_file(key, url) for key, url in self.dir_url.items() if key in to_download)

    def img_batch(self, batch_size: Optional[int] = 32,
                  target_size: Optional[Tuple] = (256, 256),
                  subset: Optional[str] = 'training',
                  validation_split: Optional[int] = 0.3,
                  class_mode: Optional[int] = 'categorical') -> keras.preprocessing.image.DirectoryIterator:
        """
        :int batch_size: size of the batches of data (default: 32)
        :tuple target_size: size of the output images.
        :str subset: `"training"` or `"validation"`.
        :float validation_split = A percentage of data to use as validation set.
        :str class_mode: Type of classification for this flow of data
            - binary:if there are only two classes
            - categorical: categorical targets,
            - sparse: integer targets,
            - input: targets are images identical to input images (mainly used to work with autoencoders),
            - None: no targets get yielded (only input images are yielded).
        """

        img_gen_params = {'featurewise_center': False,
                          'samplewise_center': True,
                          'featurewise_std_normalization': False,
                          'samplewise_std_normalization': True,
                          'zca_whitening': False,
                          'fill_mode': 'reflect',
                          'horizontal_flip': True,
                          'vertical_flip': True,
                          'validation_split': validation_split,
                          'preprocessing_function': image.rgb_to_yuv
                          }
        img_gen = keras.preprocessing.image.ImageDataGenerator(**img_gen_params)

        self.deduce_class_names()

        img_dir_params = {'directory': self.img_directory,
                          'image_data_generator': img_gen,
                          'target_size': target_size,
                          'color_mode': 'rgb',
                          'classes': self.class_names,
                          'class_mode': class_mode,
                          'batch_size': batch_size,
                          'shuffle': True
                          }
        print(subset + ':')

        return keras.preprocessing.image.DirectoryIterator(**img_dir_params, subset=subset)

    def plot_minibatch(self, sample):
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
            for Class in range(len(self.class_names)):
                if data[1][index][Class] == 1:
                    return self.class_names[i]

        plt.subplots(figsize=(20, 20))
        plt.suptitle('Batch of preprocessed images')
        # batch_size = sample[0].shape[0]
        batch_size = 4
        for i in range(batch_size):
            plt.subplot(4, batch_size // 4, i + 1)
            plt.title(str(i) + ": " + img_type(sample, i))
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(keras.preprocessing.image.array_to_img(sample[0][i]))

    def compare_img(self, size=5):
        """ Comparing images from the different classes. The files must be already downloaded.
        """
        self.deduce_class_names()

        images_folder = self.img_directory + '/'
        _, _, img_names = next(walk(images_folder + self.class_names[0]))
        img_list = list(choice(img_names, size, False))

        fig, ax = plt.subplots(nrows=len(self.class_names), ncols=size, figsize=(20, 14))

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                img = Image.open(images_folder + self.class_names[i] + '/' + img_list[j])
                col.set_axis_off()
                col.imshow(img)
                col.set_title(self.class_names[i] + " " + img_list[j])
        plt.suptitle('Display the Cover image and 3 stego images of the different algorithms')
        plt.show()
