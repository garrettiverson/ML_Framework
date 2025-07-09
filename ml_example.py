from ml_framework import ML_Framework
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

cases_dir = '/data/user/giverson/il_in' #input directory, must contain the camsim simulation folders 
out_dir = '/data/user/giverson/il_out' #output directory

ml = ML_Framework(cases_dir,out_dir, ice_layers=True) #put ice layers to true because in this example we are predicting the scattering length of 2 ice layers

ml.make_images(number_of_images=100,noise_func='default',exposure=1000,use_weights=True,conv=1.6639,gain=0,lens_model=None, from_indices=False) #this makes and saves images in output directory

# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation="relu", input_shape=(200,200,1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation="relu"),
#     layers.Flatten(),
#     layers.Dense(64, activation="relu"),
#     layers.Dense(2)  # Single output neuron for regression
# ])

model = keras.Sequential([
    layers.Flatten(input_shape=(200, 200, 1)),
    layers.Dense(128,  activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(2)  #output neuron for regression must be 2 because you are predicting scattering length of 2 ice layers
])

ml.train_NN_ttsplit(model,"adam","mse",epochs=20,metrics=["mae"]) #trains the model using a train test split, saves plots to output directory
#ml.train_NN_kfold(model,"adam","mse",epochs=10,metrics=["mae"])
ml.save_model() #saves our model in output directory
#ml.load_model(out_dir)
#images = np.load('/home/giverson/testing/il_in/scat20m_abs100m_ori270d_x0m_y1m_z-2050.2m_0/CAM_1_1_0.npy')
#print(ml.predict(images))
print(ml.test_model()) #tests the model and saves plots to output directory

