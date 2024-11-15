import os
import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


images = []
labels = []


def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:  # file
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    print(labels)
    # labels = np.array([0 if label.endswith('hhy') else 1 for label in labels])
    list = []
    for label in labels:
        if (label.endswith('boss')):
            list.append(0)
        elif (label.endswith('hhy')):
            list.append(1)
        else:
            list.append(2)
    labels = np.array([list])

    return images, labels


# Input a file path, read the images in each folder under it, and assign a different label to each folder
# Return a list of img, a list of corresponding labels, and how many folders there are (how many labels there are)

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    # IMG_SIZE = 128

    # Read all jpg files in all subfolders under the path and store them in a list
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)

        for dir_image in os.listdir(child_path):
            if dir_image.endswith('jpg'):
                # img = cv2.imread(os.path.join(child_path, dir_image))
                # resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                # recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                images = read_image(os.path.join(child_path, dir_image))
                img_list.append(images)
                label_list.append(dir_counter)

        dir_counter += 1

    # The returned img.list has been converted to np.array format
    img_list = np.array(img_list)
    label_list = np.array(label_list)
    return img_list, label_list, dir_counter


def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list