#Load libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Loading the dataset using ImageDataGenerator

image_width, image_height = 224,224
batch_size = 32

train_data_dir= '/content/drive/MyDrive/sign_language_translator/train'
test_data_dir= '/content/drive/MyDrive/sign_language_translator/test'
pred_data_dir= '/content/drive/MyDrive/sign_language_translator/pred_images'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=0.15
)

test_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='training',
                    color_mode='rgb',
                    shuffle=True,
                    target_size=(image_width, image_height)
)

validation_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='validation',
                    color_mode='rgb',
                    shuffle=False,
                    target_size=(image_width, image_height)
)

test_generator =  test_datagen.flow_from_directory(
                    test_data_dir,
                    batch_size=batch_size,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(image_width, image_height)
)
#Preprocessing for test image
pred_generator =  test_datagen.flow_from_directory(
                    pred_data_dir,
                    batch_size=batch_size,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(image_width, image_height))

#Defining the Xception
training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)

pre_trained_model=Xception(include_top=False,
                        input_shape=(image_width,image_height,3),
                        weights="imagenet")

for layer in pre_trained_model.layers:
    layer.trainable=False

#Adding fully connected layers
def initialize_model():
    model = Sequential([ pre_trained_model,
                       Flatten(),
                       Dense(128, activation='relu'),
                       #Dense(128, activation='relu'),
                       Dense(24, activation='softmax')]
       )


    return model

#Initialize model
model = initialize_model()

#Compiling model
def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = compile_model(model)

#Fitting model

es = EarlyStopping(patience=10,restore_best_weights=True)

history = model.fit(
            train_generator,
            steps_per_epoch = training_steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps_per_epoch,
            epochs=50,
            callbacks=[es]
)

#Save model
from tensorflow.keras.models import save_model
save_model(model, 'model_Xception_v3.h5')
#Evaluate model
result = model.evaluate(test_generator, verbose = 1 )
print(f'The accuracy on the test set is of {result[1]*100:.2f} %')

#Predicting
Y_pred = model.predict(test_generator,test_generator.samples / 32)
val_preds = np.argmax(Y_pred, axis=1)
val_preds

#Predicting test image
##Load model and check evaluation
xception_classifier = load_model('model_Xception_v3.h5')
res = xception_classifier.evaluate(test_generator, verbose = 1 )
print(f'The accuracy on the test set is of {res[1]*100:.2f} %')

#Preprocess test image
pred_generator =  test_datagen.flow_from_directory(
                    pred_data_dir,
                    batch_size=batch_size,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(image_width, image_height))

preds=xception_classifier.predict(pred_generator)

# create a list containing the class labels
class_labels = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
# find the index of the class with maximum score
pred = np.argmax(preds)
# print the label of the class with maximum score
class_labels[pred]
