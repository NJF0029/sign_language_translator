## IMPORTS ##
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from PIL import Image
from pylab import *
import cv2

## SKLEARN ##
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix

## TENSORFLOW/KERAS ##
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image

#########################################
#########################################

## AUGMENTATION
# TF >= 2.3.0 required
data_augmentation = models.Sequential(
    [
        #layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        layers.experimental.preprocessing.RandomContrast(factor=0.1,),
    ]
)

## INITIALIZE MODEL

def initialize_model():
    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu', padding='same'))
    data_augmentation
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Dropout to prevent over fitting
    model.add(Dropout(0.2))

    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(128, activation='relu'))


    ### Last layer - Classification Layer with 24 outputs corresponding to 24 letters in dataset
    model.add(layers.Dense(24, activation='softmax'))

    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
model = initialize_model()

## TRAIN MODEL

es = EarlyStopping(patience = 5)
save_best_cb = ModelCheckpoint(
   'models/initial-end-to-end', save_best_only = True)

history = model.fit(X_train,
                    y_train,
                    validation_split = 0.3,
                    batch_size = 32,
                    epochs = 5,
                    callbacks = [es, save_best_cb],
                    verbose = 1)

res = model.evaluate(X_test, y_test, verbose = 1 )
print(f'The accuracy on the test set is of {res[1]*100:.2f} %')

## TEST DATA - ONE HOT ENCODE
def binarize(df):
    df_labels = df['label']
    label_binarizer = LabelBinarizer()
    df_labels = label_binarizer.fit_transform(df_labels)
    return df_labels

test_labels = binarize(test)

# TRAIN DATA - RESHAPE
def reshape_data(df):

    df.drop('label', axis=1, inplace=True)

    images = df.values
    images = np.array([np.reshape(i, (28, 28)) for i in images])
    images = np.array(images)/255
    images = expand_dims(images, axis=-1)

    return images

# RESHAPE TRAIN DATA TOO??
test_images = reshape_data(test)

# MATCH LABEL TO CORRESPONDING LETTER
def get_letter(result):
    class_names = ['A',
                   'B',
                   'C',
                   'D',
                   'E',
                   'F',
                   'G',
                   'H',
                   'I',
                   'K',
                   'L',
                   'M',
                   'N',
                   'O',
                   'P',
                   'Q',
                   'R',
                   'S',
                   'T',
                   'U',
                   'V',
                   'W',
                   'X',
                   'Y' ]
    predictions=[]
    for i in y_pred:
        predictions.append(np.argmax(i))

    i = np.random.randint(1,len(predictions))
    print("Predicted Label: ", class_names[int(predictions[i])])
    print("True Label: ", class_names[int(y_test[i])])

result = np.argmax(model.predict(tf.reshape(1,28,28,1)))
get_letter(result)

#Load a jpg
#def load(new_im):
    #img = new_im
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (100,100))
    #img = img.astype('float32')/255
    #return img

#load gray reshaped image for classification
#resized_image = load(new_im)


#with open('models/intial-end-to-end-history', 'wb') as history_file:
    #pickle.dump(history.history, history_file)
#best_model = models.load_model('models/initial-end-to-end')
