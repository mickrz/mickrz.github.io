import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import csv
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Lambda, Conv2D, Cropping2D, ELU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from image_augmentation_helper import *

all_data = []
augmented_ImageData = []

print("Reading in driving info...")
center, left, right = read_driving_info_from_file("data/driving_log.csv")

print("* Filtering out minimal steering angles < abs(0.01)...")
all_data_filtered = filter_out_minimal_steering_angles(center)

print("* Adding left camera images & steering angles > 0.25 for recovery data...")
left_recovery_data = generate_left_recovery_data(left)

print("* Adding right camera images & steering angles < -0.25 for recovery data...")
right_recovery_data = generate_right_recovery_data(right)

all_data.extend(all_data_filtered)
all_data.extend(left_recovery_data)
all_data.extend(right_recovery_data)

print("Total Number of Center Camera Samples: ", len(all_data_filtered))
print("Total Number of Left Camera Samples: ", len(left_recovery_data))
print("Total Number of Right Camera Samples: ", len(right_recovery_data))
print("Total Number of Samples: ", len(all_data))

augmented_ImageData = generate_augmented_data(all_data)

print("Writing csv file...")
out = open('driving_log_extra.csv', 'w')
for row in augmented_ImageData:
    out.write('%s,%.10f\n' % (row[0], row[1]))
out.close()

def resize_images(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (64, 64))

def image_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                original_data = mpimg.imread(batch_sample[0])
                images.append(original_data)
                angles.append(float(batch_sample[1]))
            yield shuffle(np.array(images), np.array(angles))
            
Image_train, Image_validation = train_test_split(augmented_ImageData, test_size=0.20)
print("data split!")
print("training samples: ", len(Image_train))
print("validation samples: ", len(Image_validation))            

train_gen = image_generator(Image_train, batch_size=32)
validation_gen = image_generator(Image_validation, batch_size=32)


# Create modified nvidia model
model = Sequential()

model.add(Cropping2D(cropping=((70, 25), (5, 5)), dim_ordering='tf', input_shape=(160, 320, 3)))
model.add(Lambda(resize_images))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
print(model.summary())
checkpoint = ModelCheckpoint(filepath="all27_model_weights{epoch:02d}.h5", verbose=0, save_best_only=False)

model.fit_generator(train_gen, samples_per_epoch=len(Image_train), 
                    validation_data=validation_gen, nb_val_samples=len(Image_validation),
                    nb_epoch=10, callbacks=[checkpoint])

# Save off model info
model.save("model.h5")

f = open("model.json", "w")
f.write(model.to_json())
f.close()
print("model saved")