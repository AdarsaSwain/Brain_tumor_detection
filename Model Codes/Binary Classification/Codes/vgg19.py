import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.layers import Input, concatenate, Dense, Dropout
from keras.models import Model
from keras.applications import VGG19, ResNet50, InceptionV3

image_directory = './images/preprocessedData/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset = []
label = []

INPUT_SIZE = 128


for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.3, random_state=0)

# Normalize the input data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# Load pre-trained models
input_shape = (INPUT_SIZE, INPUT_SIZE, 3)
vgg_model = VGG19(weights='imagenet', input_shape=input_shape, include_top=False)
resnet_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
inception_model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)

# Freeze the layers in each model to prevent them from being trained
for layer in vgg_model.layers:
    layer.trainable = False

for layer in resnet_model.layers:
    layer.trainable = False

for layer in inception_model.layers:
    layer.trainable = False

# Combine the three models into a single model
vgg_output = vgg_model.output
vgg_output = keras.layers.GlobalAveragePooling2D()(vgg_output)

resnet_output = resnet_model.output
resnet_output = keras.layers.GlobalAveragePooling2D()(resnet_output)

inception_output = inception_model.output
inception_output = keras.layers.GlobalAveragePooling2D()(inception_output)

merged = concatenate([vgg_output, resnet_output, inception_output])
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.3)(merged)
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.2)(merged)
predictions = Dense(2, activation='softmax')(merged)

model = Model(inputs=[vgg_model.input, resnet_model.input, inception_model.input], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([x_train, x_train, x_train], y_train, epochs=5, batch_size=16)

# Evaluate the model on test data
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([x_test, x_test, x_test], y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

model.save('my_model2.h5')