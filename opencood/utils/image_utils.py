import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_files):
    img_list = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        img_list.append(image)

    #     plt.figure()
    #     plt.imshow(image)
    # plt.show()

    # exit()


    return img_list