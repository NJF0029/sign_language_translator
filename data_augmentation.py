from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

# TF >= 2.3.0
data_augmentation = models.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        layers.experimental.preprocessing.RandomContrast(factor=0.1,),
    ]
)

##
#WHERE TO ADD?
##

#def initialize_model():
    #model = models.Sequential()

    ### First Convolution & MaxPooling
    #model.add(layers.Conv2D(16, kernel_size=(4,4), input_shape=(28,28,1), activation='relu', padding='same'))
    #data_augmentation,
    #model.add(layers.MaxPool2D(pool_size=(2,2)))


########################
######## NOTES #########
########################

## TF >= 2.3.0
#data_augmentation = models.Sequential(
  #  [
 #       layers.experimental.preprocessing.RandomRotation(0.1)
  #      layers.experimental.preprocessing.RandomZoom(.1, .1),#--> look at parameters
   #     layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
    #    layers.experimental.preprocessing.RandomContrast(factor=0.1,),
   # ]
#)
