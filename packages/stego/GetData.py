import tensorflow as tf
from typing import Tuple, Optional, Generator, Dict, List

from .Stego import Stego


def augment(image, label):
    seed = None

    image = image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
    image = tf.image.per_image_standardization(image)  # standardization
    image = tf.image.random_brightness(image, max_delta=0.3, seed=seed)  # Random brightness
    image = tf.image.random_contrast(image, lower=-.1, upper=.3, seed=seed)  # Random contrast
    # image =  tf.image.random_jpeg_quality(image, min_jpeg_quality=.6, max_jpeg_quality=.75, seed=seed)
    image = tf.image.adjust_jpeg_quality(image, jpeg_quality=75, name=seed)  # adjust quality to 75
    image = tf.image.random_flip_left_right(image, seed=seed)  # Random horizontal flip
    image = tf.image.random_flip_up_down(image, seed=seed)  # Random vertical flip
    image = tf.image.rgb_to_yuv(image)

    return image, label


class GetData(Stego):
    def __init__(self, dir_url: Dict[str, str] = None, class_names: [List] = None, input_directory: str = '/content',
                 subdirectory: str = '/stego_images'):
        super().__init__(dir_url, class_names, input_directory, subdirectory)
        self.dir_url = dir_url
        self.class_names = class_names
        self.img_directory = input_directory + subdirectory

    def download_unzip(self, get_all=True) -> None or Generator:
        """
        Downloads data from the dir_url of the form {category, url}.
        Stores each folder under input_directory/subdirectory/category/

        PARAMETERS:
        :param get_all: [default=True] Will automatically start downloading all the files in the values of
         "self.dir_url" to store them in folders with the keys of that directory.

        :return:
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
                tf.keras.utils.get_file(**f_params)
        else:
            return (tf.keras.utils.get_file(key, url) for key, url in self.dir_url.items() if key in to_download)

    def img_batch(self, batch_size: Optional[int] = 32,
                  target_size: Optional[Tuple] = (256, 256),
                  subset: Optional[str] = 'training',
                  validation_split: Optional[int] = 0.3,
                  class_mode: Optional[int] = 'categorical',
                  preprocessing_function=None) -> tf.keras.preprocessing.image.DirectoryIterator:
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
                          'preprocessing_function': preprocessing_function
                          }
        img_gen = tf.keras.preprocessing.image.ImageDataGenerator(**img_gen_params)

        self._deduce_class_names()

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

        return tf.keras.preprocessing.image.DirectoryIterator(**img_dir_params, subset=subset)
