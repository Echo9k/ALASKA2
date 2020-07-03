from numpy.random import choice
from os import walk
from typing import Dict, List
import matplotlib.pyplot as plt

from PIL import Image


class Stego:
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

    def compare_img(self, size=5):
        """ Comparing images from the different classes."""
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
