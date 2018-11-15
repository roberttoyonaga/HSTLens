
# coding: utf-8

# In[2]:



from HSTLens_base_classifier_resnet13_CC import BaseKerasClassifier
from HSTLens_blocks_resnet13_CC import pre_activated_resnet_block

from keras.layers import Activation, AveragePooling2D, MaxPooling2D
from keras.layers import Conv2D, ELU, Dropout

from keras.layers.normalization import BatchNormalization

class deeplens_classifier(BaseKerasClassifier):
    '''
    def __init__(self, **kwargs):
        """
        Initialisation
        """
        super(self.__class__, self).__init__(**kwargs)
    '''
    def _model_definition(self, net):
        """
        Builds the architecture of the network
        """
        
        # Input filtering and downsampling with max pooling
        print(net.shape)  #channels must be specified first otherwise keras assumes channels last
        
        '''    
        net = Conv2D( filters=256, kernel_size=9, activation='relu', padding='same',
                     data_format="channels_first", input_shape=(1, 100, 100))(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
        net = Conv2D( filters=128, kernel_size=7, activation='relu', padding='same', data_format="channels_first")(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
        net = Conv2D( filters=64, kernel_size=5, activation='relu', padding='same', data_format="channels_first")(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
        net = Conv2D( filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_first")(net)
        net= MaxPooling2D(pool_size=(2,2))(net)
 
        ''' 
            
        
        '''
        # Input filtering and downsampling with max pooling
        print(net.shape)  #channels must be specified first otherwise keras assumes channels last
        net = Conv2D( filters=32, kernel_size=3, activation='relu', padding='same',
                     data_format="channels_first", input_shape=(4, 101, 101))(net)
        #net = BatchNormalization()(net)
        print(net.shape)
        '''
        
        print("resnet_14")
        
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32,  preactivated=False)
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64,)
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128,)
        '''
       
        net = pre_activated_resnet_block(net, n_filters_in=128, n_filters_out=64,  preactivated=False, kernel=7,)
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=32,kernel=5,)
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=16, kernel=3,)
        
        '''

        '''
        pool_size = net.output_shape[-1]
        net= AveragePooling2D(pool_size=pool_size, strides=1)
        '''
        
        
        
        #pool_size = net.output_shape[-1]
        #net= AveragePooling2D(pool_size=(2,2), strides=1)
        
    
        return net


# In[ ]:




