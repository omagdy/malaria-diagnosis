import numpy as np
import tensorflow as tf

"""
 Data Augmentation is done by randomly flipping every 
 image around both the y and x axes before using them for training.
"""

def flip_image_x(image):
    return image[::-1, :]

def flip_image_y(image):
    return image[:, ::-1]

def get_image_batch(data, idx, BATCH_SIZE, r_x=0, r_y=0):
    sub_data = data[idx:idx+BATCH_SIZE]
    for i in range(BATCH_SIZE):
        if r_x == 1:
            sub_data[i] = flip_image_x(sub_data[i]) # Data Augmentation
        if r_y == 1:
            sub_data[i] = flip_image_y(sub_data[i]) # Data Augmentation
    return sub_data

def get_label_batch(data, idx, BATCH_SIZE):
    sub_data = data[idx:idx+BATCH_SIZE].reshape(-1,1)
    return sub_data


# Normalize data so that model doesnt get large inputs and converge faster
def NormalizeData(data, min_value=0, max_value=255):
    return (data-min_value)/(max_value-min_value)


def data_preprocessing(malaria_images, no_malaria_images):
    
    IMAGE_SIZE = 40

    # Initializing empty arrays for each class (malaria or non-malaria) and populate it with it with data
    numpy_arrays    = []
    all_image_files = [malaria_images, no_malaria_images]
    for image_files in all_image_files:
        numpy_array = np.empty((0,IMAGE_SIZE,IMAGE_SIZE,3), 'uint8')
        for file in image_files:
            image_contents = tf.io.read_file(file)
            image = tf.image.decode_jpeg(image_contents, channels=3).numpy()
            if image.shape == (IMAGE_SIZE,IMAGE_SIZE,3):
                numpy_array = np.append(numpy_array, [image] , axis=0)
        numpy_arrays.append(numpy_array)
    malaria_images_data, no_malaria_images_data = numpy_arrays

    malaria_images_data = NormalizeData(malaria_images_data)
    no_malaria_images_data = NormalizeData(no_malaria_images_data)

    # Create label data for each class
    malaria_images_label    = np.ones(malaria_images_data.shape[0])
    no_malaria_images_label = np.zeros(no_malaria_images_data.shape[0])

    data   = np.append(malaria_images_data, no_malaria_images_data, axis=0)
    labels = np.append(malaria_images_label, no_malaria_images_label)

    return data, labels
