from camsim import sim as Sim
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
import json 
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import keras_nlp


####################################### KERAS EXAMPLE ViT from https://keras.io/examples/vision/image_classification_with_vision_transformer/ #############################

#multilayer perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # Use static shapes for height, width, channels
        height = images.shape[1]
        width = images.shape[2]
        channels = images.shape[3]
        
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        #patches = tf.image.extract_patches(images, size=self.patch_size)
        patches = tf.image.extract_patches( images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        
        patches = tf.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def create_vit_classifier(img_size = 500, patch_size=20,num_classes=1,projection_dim = 64,transformer_layers = 3,num_heads = 8,mlp_head_units = [2048,1024],transformer_units = None):
    
    if transformer_units is None:
        transformer_units = [projection_dim * 2,projection_dim]
    
    num_patches = (img_size // patch_size) ** 2
    inputs = keras.Input(shape=(img_size,img_size,1))
    # Augment data.
    #augmented = data_augmentation(inputs)
    augmented = inputs
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

##################################################### OPTICUS ViT ########################################################################################
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dense(embed_dim)
        ])
        
    def call(self, x):
        x_att = self.att(x, x)
        x = self.norm1(x + x_att)
        x_mlp = self.mlp(x)
        x = self.norm2(x + x_mlp)
        return x

class ViT(Model):
    def __init__(self,
                 img_size=500,
                 patch_size=50,
                 embed_dim=256,
                 depth=3,
                 num_heads=8,
                 mlp_dim=512,
                 num_classes=1):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (img_size // patch_size) ** 2
        self.seq_len = num_patches + 1

        # Patch embedding: Conv2D
        self.patch_embed = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')

        # Learnable class token and positional embedding
        self.cls_token = self.add_weight("cls_token",
                                         shape=(1, 1, embed_dim),
                                         initializer=initializers.TruncatedNormal(stddev=0.02),
                                         trainable=True)

        self.pos_embed = self.add_weight("pos_embed",
                                         shape=(1, self.seq_len, embed_dim),
                                         initializer=initializers.TruncatedNormal(stddev=0.02),
                                         trainable=True)

        # Transformer encoder layers
        self.transformer_layers = [
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ]

        # Output head
        self.head = layers.Dense(num_classes,
                                 kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                                 bias_initializer='zeros')

    def call(self, inputs):
        """
        inputs: shape (B, img_size, img_size, 1)
        """
        B = tf.shape(inputs)[0]

        # (1) Patch Embedding
        x = self.patch_embed(inputs)  # (B, 10, 10, embed_dim)

        # (2) Flatten & Transpose to (B, num_patches, embed_dim)
        x = tf.reshape(x, [B, -1, x.shape[-1]])  # (B, 100, embed_dim)

        # (3) Add CLS token
        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, x.shape[-1]])
        x = tf.concat([cls_tokens, x], axis=1)  # (B, 101, embed_dim)

        # (4) Add positional embedding
        x = x + self.pos_embed

        # (5) Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # (6) Use CLS token output
        cls_out = x[:, 0]  # (B, embed_dim)

        # (7) Regression head
        out = self.head(cls_out)  # (B, 1)
        return tf.squeeze(out, axis=-1)  # (B,)








########################################################### Noise Model ################################################################################################
#experimental found values for noise model
offset = -238.3010098283891
amplitude = 0.8593836419304328
exponent = 0.5148938848756486

def noise_model(average):

    if average > (-240.0 - offset):

        return 1 * amplitude * np.power(average+240.+offset,exponent)

    else:

        return 1 *amplitude*0.1 
        
def noise_randomizer(average):

    return int(np.random.normal(average,noise_model(average),1)[0])
    
noisemodel = np.vectorize(noise_randomizer)

####################################################################### MACHINE LEARNING FRAMEWORK ################################################################

class ML_Framework:
    '''MLFramework is a class for training a neural network on images from simulations produced by CamSim.'''
    def __init__(self,cases_dir, out_dir,variables_list='default',num_camera=(1,1,0), ice_layers=False, variable_dict='default'):
        '''MLFramework(case_dir,out_dir,variables_list='default',num_camera=(1,1,0), ice_layers=False, variable_dict='default'), where cases_dir is the directory 
        filled with the different simulation folders output from camsim (expected to already have finished simulations),
        out_dir is the directory where you want the output from this class to be, variables list
        is a list of strings for the variables you want to use in your analysis named the same as the config.json file saved from camsim, 
        the default will check anything different among cases, num_camera is a tuple describing
        the camera in the simulation you want to take images from in form (string #, OM #, index of camera on DOM). ice_layers
        if set to true will mean that you are trying to predict scattering length of two ice layers (make sure only 1 ice layers scattering length
        is different from all the rest), if ice_layers is set to false it will be assumed that scattering length is the same everywhere 
        for your cases and will only predict the one value if scattering lengths are different across the cases. variable_dict
        is the dictionary mapping simulation folder names to their list of values for each variable that are different. Leaving this as 
        default means it will find set up this dictionary for you, if you want to use your own values pass in the dictionary and also
        the variable names in variables_list.'''
        
        self.cases_dir = cases_dir #directory of the different simulations
        if self.cases_dir[-1] != '/':
            self.cases_dir = cases_dir + '/'
            
        self.out_dir = out_dir #output directory
        if self.out_dir[-1] != '/':
            self.out_dir = out_dir + '/'

        self.variables_list = variables_list #list of variables to look at between cases
        
        self.num_camera = num_camera #store the camera number
        
        self.simfolders = [x for x in os.listdir(self.cases_dir) if os.path.isdir(self.cases_dir + '/' + x) and x != '.ipynb_checkpoints'] #the simulation folders
        
        self.simulations = [Sim.Simulator.load_folder(self.cases_dir + folder) for folder in self.simfolders] #the simulation objects in each folder
        self.ice_layers = ice_layers
       
        self.configs = []  #store list of all config files
        for simfolder in self.simfolders:
            
            self.configs.append((json.load(open(self.cases_dir + simfolder + '/config.json','r')),simfolder))

        self.variables = [] #storing the variables that are different from each other
        self.values = {} #dictionary storing 'case name': value of variables
        self.modellist = [] #list of variables names that are different between cases
        self.model = None #the machine learning model
        self.mean = None #the mean used to normalize the dataset
        self.std = None #the stddev used to normalize the dataset
        
        if variables_list == 'default':
            self.variables_list = ['Absorption','Scattering','hole_scatt',"hole_radius",'beam_width', 'orientation', 'tilt', 'position','hole_offset']
            
        if variable_dict != 'default':
            self.values = variable_dict
            self.modelllist = self.variables_list
        else:
            #Lets check for differences between the cases 
            #first for non-camera, non-LED parameters
            for check in ['Absorption','Scattering']:
                if check in self.variables_list:
                    initial = self.configs[0][0][check]
                    for i in self.configs:
    
                        if not i[0][check] == initial:
    
                            self.variables.append(('cfg',check))
            
            
            #General hole parameters
            for check in ['hole_scatt',"hole_radius"]:
                if check in self.variables_list:
                    initial = self.configs[0][0][check]
                    for i in self.configs:
    
                        if not i[0][check] == initial:
    
                            self.variables.append(('cfg',check))
    
            #now the LED parameters
            for check in ['beam_width', 'orientation', 'tilt', 'position','hole_offset']:
                if check in self.variables_list:
                    initial = self.configs[0][0]['Emitters'][0][check]
                    
                    for i in self.configs:
    
                        if not i[0]['Emitters'][0][check] == initial:
                            #added all this so different dimensions are treated seperately
                            if check == 'position':
                                if not i[0]['Emitters'][0][check][0] == initial[0]:
                                    self.variables.append(('Emitters',check + '_x'))
                                if not i[0]['Emitters'][0][check][1] == initial[1]:
                                    self.variables.append(('Emitters',check + '_y'))
                                if not i[0]['Emitters'][0][check][2] == initial[2]:
                                    self.variables.append(('Emitters',check+'_z'))
                            else:
                                self.variables.append(('Emitters',check))
    
            #now the cameras
            for check in ['position','hole_offset']:
                if check in self.variables_list:
                    initial = self.configs[0][0]['Cameras'][0][check]
                    for i in self.configs:
                        
                        if not i[0]['Cameras'][0][check] == initial:
                            if check == 'position':
                                if not i[0]['Cameras'][0][check][0] == initial[0]:
                                    self.variables.append(('Cameras',check + '_x'))
                                if not i[0]['Cameras'][0][check][1] == initial[1]:
                                    self.variables.append(('Cameras',check + '_y'))
                                if not i[0]['Cameras'][0][check][2] == initial[2]:
                                    self.variables.append(('Cameras',check+'_z'))
                            else:
                                self.variables.append(('Cameras',check))
    
            #finally lets store the different variables and their values in modeldict
            uniq_variables = set(self.variables)
    
            for case in self.simfolders:
                y = [x for x in self.configs if x[1] == case]
    
                y_val = []
    
                for variable in uniq_variables:
                    
                    if 'Emitters' in variable or 'Cameras' in variable:
                        if 'position' in variable[1]:
                            if '_x' in variable[1]:
                                y_val.append(y[0][0][variable[0]][0][variable[1][:-2]][0])
                            if '_y' in variable[1]:
                                y_val.append(y[0][0][variable[0]][0][variable[1][:-2]][1])
                            if '_z' in variable[1]:
                                y_val.append(y[0][0][variable[0]][0][variable[1][:-2]][2])
                        else:
                            y_val.append(y[0][0][variable[0]][0][variable[1]])
                    else:
                        if not isinstance(y[0][0][variable[1]],list):
                            y_val.append(y[0][0][variable[1]])
                        else:
                            #scattering or absorption is a list here
                            if self.ice_layers:
                                value = list(np.unique(y[0][0][variable[1]]))
                                for val in value:
                                    y_val.append(val)
                                if len(value) < 2:
                                   y_val.append(value[0])
                            else:
                                value = y[0][0][variable[1]][0]
                                y_val.append(value)
                          
                
                self.values.update({case:y_val})
            self.modellist = [x[1] for x in uniq_variables]
            

    def make_images(self,number_of_images=100,noise_func='default',exposure=1000,use_weights=True,conv=1.6639,gain=0,lens_model=None,test_size=0.1, val_size=0.2, from_indices=False):
        '''make_images(number_of_images=100,noise_func='default',exposure=1000,use_weights=True,conv=1.6639,gain=0,lens_model=None,
        test_size=0.1, val_size=0.2, from_indices=False): where number_of_images is the number of images to generate, exposure is the exposure time in ms, 
        use_weights is set to true if we want to use pixel counts instead of  
        photon counts,conv is the conversion factor to use when converting photon counts 
        to pixel values,gain is the gain of the sensor,
        lens_model=None uses the camera model already defined but can also 
        have values 'default' which means cameras parameters will be automatically 
        set based on resolution or can use a json file or choose 'pinhole' for a pinhole camera, 
        noise_func is the noise function to use to add noise to the images, test_size 
        is the percentage of images made that will be held out for a final test used in the test_model function.
        val_size is the percentage of images that will be used for validation if you use the train test split function. 
        from_indices means that you used grid computing and the combine_images function to make a final images .npy file, this will 
        use that file instead of your lower statistic images, don't use noise when using combine_images. Makes a dataset of images that 
        other functions will use to train/validate/test your model, they will be saved in your output directory. Only have to use once then can
        train on this same set of images many different models. This function also normalizes the data set based on the training set.'''

        images_dict = {}
        
        if noise_func == 'default':
            noise_func = noisemodel
            
        if not from_indices:
            for i,simulation in enumerate(self.simulations):
                simulation.camera_make_images(self.num_camera[0],self.num_camera[1],self.num_camera[2], exposure=exposure, gain = gain, use_weights = use_weights, lens_model = None, conversion = conv, number_of_images = number_of_images,cmap = 'jet',noise_func=noise_func) 
                
                images_dict.update({self.simfolders[i] : 
                                    np.load(self.cases_dir + self.simfolders[i]+'/CAM_'+str(self.num_camera[0])+'_'+str(self.num_camera[1])+'_'+str(self.num_camera[2])+'.npy')})
                
        else:
            for i,simfolder in enumerate(self.simfolders):
                cam_name = 'final_'+str(self.num_camera[0])+'_'+str(self.num_camera[1])+'_'+str(self.num_camera[2])+'.npy'
                if cam_name in os.listdir(self.cases_dir + simfolder):
                    template = np.load(self.cases_dir + simfolder + '/' + cam_name)
                    images = []
                    for n in range(number_of_images):
                        images.append(noise_func(template))
                    images_dict.update({simfolder: np.array(images)})
                else:
                    continue
                
        #create and store lists of images and truth values
        x, y = [], []
        for folder in self.simfolders:
            if folder not in images_dict:
                continue
            image_list = images_dict[folder]
            x.append(image_list)
            y += [self.values[folder]] * len(image_list)
        
        x = np.array(np.concatenate(x))
        y = np.array(y)
        #and finally add channel at end
        x = x[..., np.newaxis]

        # Shuffle
        perm = np.random.permutation(len(y))
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        # Compute split index
        split_index1 = int(len(x_shuffled) * (1 - test_size - val_size))
        split_index2 = int(len(x_shuffled)*(1-test_size))
        # Correct train-test split
        x_train = x_shuffled[:split_index1]
        x_val = x_shuffled[split_index1:split_index2]
        x_test  = x_shuffled[split_index2:]
        y_train = y_shuffled[:split_index1]
        y_val = y_shuffled[split_index1:split_index2]
        y_test  = y_shuffled[split_index2:]
        
        #normalize x 
        self.mean = np.mean(x_train)
        self.std = np.std(x_train)
        x_train = (x_train - self.mean) / self.std
        x_val = (x_val - self.mean) / self.std
        x_test = (x_test - self.mean) / self.std
        # Save
        np.save(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}train_x.npy', x_train)
        np.save(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}train_y.npy', y_train)
        np.save(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}val_x.npy', x_val)
        np.save(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}val_y.npy', y_val)
        np.save(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}test_x.npy', x_test)
        np.save(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}test_y.npy', y_test)
        np.save(self.out_dir+'norm_constants.npy',np.array([self.mean, self.std]))
            
    def train_NN_ttsplit(self,model, optimizer, loss, epochs=10, batch_size=32, metrics=None, plot=True, plot_metrics=True):
        '''train_NN_ttsplit(model,optimizer,loss,epochs=10,batch_size=32,metrics=None, plot=True,plot_metrics=True),model is a keras 
        neural network model (make sure that your final layer is a Dense layer with size equal to the number of variables/classes you are trying to predict and 
        input_shape = (resolution.x, resolution.y,1) for the first layer), optimizer is the name of the optimizer you want to use from keras, 
        loss is the name of the loss you want to use from keras or your own defined loss function which 
        must work on tensorflow tensors output must be a scalar averaged or summed over the batch, epochs
        is the number of times to iterate completely over the dataset during training, batch_size is the number of images to train on at once, 
        metrics is a list of metrics from keras you want to evaluate your model with, plot is set to true to plot the loss over 
        epochs, plot_metrics is set to true to plot the metrics you chose over the epochs.
        Given a neural network keras model it will train on the images in output directory. Returns a keras history object.'''
        X_train, X_val = np.array([]), np.array([])
        y_train, y_val = np.array([]), np.array([])
        try:
            self.mean, self.std = np.load(self.out_dir+'norm_constants.npy')
            X_train = np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}train_x.npy')
            X_val = np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}val_x.npy')
            y_train = np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}train_y.npy')
            y_val = np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}val_y.npy')
        except e:
            print(e)
            print("No images found, run make_images first.")
            return
            
        if metrics is None:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

        #plotting of loss, predictions for validation set, and metrics
        if plot:
            #plot loss
            plt.clf()
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.out_dir+'loss_curve.png')
            
        #plot metrics
        if plot_metrics and not metrics is None:
            for metric in metrics:
                plt.clf()
                plt.plot(history.history[metric], label='Training '+metric)
                plt.plot(history.history['val_'+metric], label='Validation '+metric)
                plt.title(metric+' Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True)
                plt.savefig(self.out_dir+metric+'_curve.png')

        self.model = model
        return history

    
    def train_NN_kfold(self,model, optimizer, loss, epochs=10, batch_size=32, metrics=None, k=5, shuffle=True, random_state=1,plot=True, plot_metrics=True):
        '''train_NN_ttsplit(model,optimizer,loss,epochs=10,batch_size=32,metrics=None,test_size=0.2,random_state=1,plot=True,plot_metrics=True), model is a keras 
        neural network model (make sure that your final layer is a Dense layer with size equal to the number of variables you are trying to predict and 
        input_shape = (resolution.x, resolution.y,1) for the first layer), optimizer is the name of the optimizer you want to use from keras, 
        loss is the name of the loss you want to use from keras or your own defined loss function which must work on tensorflow tensors output 
        must be a scalar averaged or summed over the batch, epochs is the number of times to iterate completely over the dataset during 
        training, batch_size is the number of images to train on at once, 
        metrics is a list of metrics from keras you want to evaluate your model with, k is the number of folds to use for the kfold cross validation,
        shuffle is true means the data in each fold is randomly selected, random_state is the random state of the selected folds, plot is set 
        to true to plot the mean loss (over folds) vs epochs, plot_metrics is set to true to plot the mean metrics you chose (over folds) over the epochs.
        Trains a given model using kfold cross validation (keeps trained the model on the last fold) using both training and validation data
        and renormalizes the training and validation data based on the training data of each fold. No model is saved.
        Returns a dictionary containing the final loss and metrics mean over the k folds and also the standard deviation of 
        the loss and metrics over the k folds.'''

        images = np.array([])
        truths = np.array([])
        try:
            self.mean, self.std = np.load(self.out_dir+'norm_constants.npy')
            images = np.concatenate((np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}train_x.npy'),
                                     np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}val_x.npy')))
            truths = np.concatenate((np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}train_y.npy'),
                                     np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}val_y.npy')))
            #lets get them un-normalized
            images = (images*self.std) + self.mean
            
        except Exception as e:
            print(e)
            print("No images found, run make_images first.")
            return
            
            
        train_losses = []
        val_losses = []
        metric_train_values = [[] for m in metrics]
        metric_val_values = [[] for m in metrics]
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        
        if metrics is None:
            for fold, (train_index, val_index) in enumerate(kf.split(images)):
                print(f'Fold {fold + 1}')
                
                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = images[train_index], images[val_index]
                y_train_fold, y_val_fold = truths[train_index], truths[val_index]

                #Normalize over fold
                fold_mean = np.mean(X_train_fold)
                fold_std = np.std(X_train_fold)
                X_train_fold = (X_train_fold - fold_mean)/fold_std
                X_val_fold = (X_val_fold - fold_mean)/fold_std
                
                # Create a new instance of the model for each fold
                model.compile(optimizer=optimizer, loss=loss)
                
                # Train the model on the training fold
                history = model.fit(X_train_fold, y_train_fold,validation_data=(X_val_fold,y_val_fold), epochs=epochs, batch_size=batch_size, verbose=1)
                
                train_losses.append(history.history['loss'])
                val_losses.append(history.history['val_loss'])
                
        else:
            for fold, (train_index, val_index) in enumerate(kf.split(images)):
                print(f'Fold {fold + 1}')
                
                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = images[train_index], images[val_index]
                y_train_fold, y_val_fold = truths[train_index], truths[val_index]
                
                #Normalize over fold
                fold_mean = np.mean(X_train_fold)
                fold_std = np.std(X_train_fold)
                X_train_fold = (X_train_fold - fold_mean)/fold_std
                X_val_fold = (X_val_fold - fold_mean)/fold_std
                
                # Create a new instance of the model for each fold
                model.compile(optimizer=optimizer, loss=loss,metrics=metrics)
                
                # Train the model on the training fold
                history = model.fit(X_train_fold, y_train_fold,validation_data=(X_val_fold,y_val_fold), epochs=epochs, batch_size=batch_size,verbose=1)
                train_losses.append(history.history['loss'])
                val_losses.append(history.history['val_loss'])
                for i,metric in enumerate(metrics):
                    metric_train_values[i].append(history.history[metric])
                    metric_val_values[i].append(history.history['val_'+metric])
                    
        # Convert and average
        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        metric_train_values = np.array(metric_train_values)
        metric_val_values = np.array(metric_val_values)
        
        mean_train_loss = np.mean(train_losses, axis=0)
        mean_val_loss = np.mean(val_losses, axis=0)
        mean_metric_train = [np.mean(m,axis=0) for m in metric_train_values]
        mean_metric_val = [np.mean(m,axis=0) for m in metric_val_values]

            
        #plotting of loss and metrics
        if plot:
            plt.clf()
            plt.plot(mean_train_loss, label='Avg Training Loss')
            plt.plot(mean_val_loss, label='Avg Validation Loss')
            plt.title(f'{k}-Fold Cross-Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.out_dir+'loss_curve.png')
            
        
        if plot_metrics and not metrics is None:
            for i,metric in enumerate(metrics):
                plt.clf()
                plt.plot(mean_metric_train[i], label='Avg Training '+metric)
                plt.plot(mean_metric_val[i], label='Avg Validation '+metric)
                plt.title(f'{k}-Fold Cross-Validation {metric}')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True)
                plt.savefig(self.out_dir+metric+'_curve.png')

        #create dictionary of important stats and return it
        #get mean loss of last epoch for all folds, and also standard deviation of the losses of the last epoch across the k folds
        stats = {"train_loss" : [mean_train_loss[-1],np.std(train_losses[:,-1],ddof=1)],
                "val_loss": [mean_val_loss[-1],np.std(val_losses[:,-1],ddof=1)]}
        
        #get mean loss of last epoch for all folds, and also standard deviation of the losses of the last epoch across the k folds
        for i,metric in enumerate(metrics):
            stats['train_'+metric] = [mean_metric_train[i][-1],np.std(metric_train_values[i][:,-1],ddof=1)]
            stats['val_'+metric] = [mean_metric_val[i][-1],np.std(metric_val_values[i][:,-1],ddof=1)]
            
        return stats
        
    def save_model(self):
        '''save_model(), saves the most recently trained model (that has used train test split) using keras.'''
        #check if we have made a model yet
        if self.model is None:
            return
        #save the model
        self.model.save(self.out_dir)

    def predict(self,ims):
        '''predict(ims), ims is a list of images to predict, they shouldn't have a channel axis and shouldn't be normalized. 
        If you pass one image make sure to surround it in an array. 
        Uses the most recently trained model to predict parameter values from the images, returns those values. '''
        #check if we have made a model yet
        if self.model is None:
            return
            
        #if we don't have channel dimension add that
        if ims.shape[-1] != 1:
            ims = ims[..., np.newaxis]
            
        #normalize
        normalized_images = (ims - self.mean)/self.std
        
        #return prediction
        return self.model.predict(normalized_images)
        
    def test_model(self):
        '''test_model(): This function uses held out generated images from the output directory 
        and the most recently trained model to predict the parameter values, 
        it returns a dictionary of the loss and metrics and their values on the 
        newly generated dataset. '''
        #check if we have made a model yet
        if self.model is None:
            return

        x_test = np.array([])
        y_test = np.array([])
        try:
            x_test = np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}test_x.npy')
            y_test = np.load(f'{self.out_dir}CAM_{self.num_camera[0]}_{self.num_camera[1]}_{self.num_camera[2]}test_y.npy')
            #print("TEST SIZE: ", len(x_test))
        except Exception as e:
            print(e)
            print("No images found, run make_images first.")
            return
        
        y_pred = self.model.predict(x_test)
        #if its a 1d array fix it
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1,1)

        #plot each variable and prediction
        for d in range(len(y_pred[0])):
            plt.clf()
            
            plt.scatter(y_test[:,d],y_pred[:,d])            
            if self.ice_layers:
                index = d
                #It is possible to have the same variable twice for scattering of ice layers
                if len(self.modellist)-1 < d:
                    index = len(self.modellist)-1
                variable_name = self.modellist[index]
                plt.ylabel(f'Predicted {variable_name}{d}')
                plt.xlabel(f'Actual {variable_name}{d}')
                min_value = np.min(np.concatenate((y_test[:,d],y_pred[:,d])))
                max_value = np.max(np.concatenate((y_test[:,d],y_pred[:,d])))
                plt.xlim([min_value,max_value])
                plt.ylim([min_value,max_value])
                plt.plot([min_value, max_value], [min_value, max_value], 'r--')
                plt.title(f'Predicted vs Actual {variable_name}{d}')
                plt.savefig(self.out_dir+variable_name+str(d)+'_test.png')
                
            else:
                variable_name = self.modellist[d]
                plt.ylabel(f'Predicted {variable_name}')
                plt.xlabel(f'Actual {variable_name}')
                min_value = np.min(np.concatenate((y_test[:,d],y_pred[:,d])))
                max_value = np.max(np.concatenate((y_test[:,d],y_pred[:,d])))
                plt.xlim([min_value,max_value])
                plt.ylim([min_value,max_value])
                plt.title(f'Predicted vs Actual {variable_name}')
                plt.plot([min_value, max_value], [min_value, max_value], 'r--')
                plt.savefig(self.out_dir+variable_name+'_test.png')
        
        #return a dictionary of loss and metrics used in training
        return dict(zip(self.model.metrics_names, self.model.evaluate(x_test, y_test, verbose=1))) 

    def get_model(self):
        '''get_model(), returns the most recently trained model.'''
        if self.model is None:
            print("No model to get")
            return
        return self.model
        
    def load_model(self, directory):
        '''load_model(directory), loads a keras model and normalization constants from a directory'''
        norm_array = np.load(self.out_dir+'norm_constants.npy')
        self.mean = norm_array[0]
        self.std = norm_array[1]
        self.model = keras.models.load_model(directory)
        
    def get_normalization(self):
        '''returns constants used for normalization of image dataset'''
        return np.array([self.mean,self.std])
        
    def get_ViT(self,img_size=500,
                 patch_size=50,
                 embed_dim=256,
                 depth=3,
                 num_heads=8,
                 mlp_dim=512,
                 num_classes=1):
        '''
        returns a keras model vision transformer with architecture similar to OPTICUS (created by Minje Park)
        '''
        return ViT(img_size=img_size,
                 patch_size=patch_size,
                 embed_dim=embed_dim,
                 depth=depth,
                 num_heads=num_heads,
                 mlp_dim=mlp_dim,
                 num_classes=num_classes)
        
    def get_ViT2(self,img_size = 500, 
                 patch_size=20,
                 num_classes=1,
                 projection_dim = 64,
                 transformer_layers = 3,
                 num_heads = 8,
                 mlp_head_units = [2048,1024],
                 transformer_units = None):
        '''
        returns a keras model vision transformer from a keras example here: https://keras.io/examples/vision/image_classification_with_vision_transformer/
        '''
        return create_vit_classifier(img_size = img_size, 
                                     patch_size=
                                     patch_size,
                                     num_classes=num_classes,
                                     projection_dim = projection_dim,
                                     transformer_layers = transformer_layers,
                                     num_heads = num_heads, 
                                     mlp_head_units = mlp_head_units, 
                                     transformer_units = transformer_units)
    
            
        
        
                
    
