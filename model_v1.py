#Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.backend import expand_dims
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#Load training and test data
train = pd.read_csv('/Users/nathan/Desktop/kaggle_sign_mnist/sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('/Users/nathan/Desktop/kaggle_sign_mnist/sign_mnist_test/sign_mnist_test.csv')

print(train.shape)
print(test.shape)

#One hot encode train label
def binarize(df):
    df_labels = df['label']
    label_binarizer = LabelBinarizer()
    df_labels = label_binarizer.fit_transform(df_labels)
    return df_labels

train_labels_encoded = binarize(train)

#Reshape train data
def reshape_data(dataframe):

    dataframe.drop('label', axis=1, inplace=True)

    images = dataframe.values
    images = np.array([np.reshape(i, (28, 28)) for i in images])
    #images = expand_dims(images, axis=-1)

    return images

train_images = reshape_data(train)
train_images

print(type(train_labels_encoded))

#train test split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_encoded, test_size=0.3, random_state=42)

# Scale images
X_train = np.array(X_train) / 225
X_test = np.array(X_test) / 225

#Creating function that will initialise our model
def initialize_model():
    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(24, activation='relu'))

    ### Last layer - Classification Layer with 10 outputs corresponding to 24 letters in dataset
    model.add(layers.Dense(24, activation='softmax'))

    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

#Train our model
from tensorflow.keras.callbacks import EarlyStopping

model = initialize_model()

es = EarlyStopping(patience = 5)

history = model.fit(X_train,
                    y_train,
                    validation_split = 0.3,
                    batch_size = 32,
                    epochs = 5,
                    callbacks = [es],
                    verbose = 1)

res = model.evaluate(X_test, y_test, verbose = 1 )
print(f'The accuracy on the test set is {res[1]*100:.2f} %')

test_labels = binarize(test)
test_images = reshape_data(test)

#Evaluate performance on unseen data
y_pred = model.predict(test_images)

#Accuracy score on unseen test data
print(f'The accuracy of the model on the unseen test data is {accuracy_score(test_labels, y_pred.round())*100:.2f} %')
