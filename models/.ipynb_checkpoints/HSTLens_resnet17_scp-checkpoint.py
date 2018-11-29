
# coding: utf-8

# In[2]:



from HSTLens_base_classifier_resnet17_s import BaseKerasClassifier

from keras.layers import Activation, AveragePooling2D, MaxPooling2D
from keras.layers import Conv2D, ELU, Dropout, LeakyReLU

from keras.layers.normalization import BatchNormalization

class deeplens_classifier(BaseKerasClassifier):

    def _model_definition(self, net):
        """
        Builds the architecture of the network
        """
        
        # Input filtering and downsampling with max pooling
        print(net.shape)  #channels must be specified first otherwise keras assumes channels last
        print('resnet17_scp')
  
        net = Conv2D( filters=128, kernel_size=5, activation=None, padding='same', 
                     data_format="channels_first", input_shape=(1, 100, 100))(net)
        net = BatchNormalization(axis=1)(net) #axis is set to the dimension which hold the colour channels
        net = LeakyReLU()(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
        
        net = Conv2D( filters=64, kernel_size=3, activation=None, padding='same', data_format="channels_first")(net)
        net = BatchNormalization(axis=1)(net) #axis is set to the dimension which hold the colour channels
        net = LeakyReLU()(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
        
        net = Conv2D( filters=64, kernel_size=3,activation=None, padding='same', data_format="channels_first")(net)
        net = BatchNormalization(axis=1)(net) #axis is set to the dimension which hold the colour channels        
        net = LeakyReLU()(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
        


    
        return net


# In[ ]:




