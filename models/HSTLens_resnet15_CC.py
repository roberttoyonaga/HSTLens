
# coding: utf-8

# In[2]:



from HSTLens_base_classifier_resnet15_CC import BaseKerasClassifier
from HSTLens_blocks_resnet15_CC_BN import pre_activated_resnet_block

from keras.layers import Activation, AveragePooling2D, MaxPooling2D
from keras.layers import Conv2D, ELU, Dropout

from keras.layers.normalization import BatchNormalization

class deeplens_classifier(BaseKerasClassifier):

    def _model_definition(self, net):
        """
        Builds the architecture of the network
        """
        
        
      
        
        
        '''
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32,  preactivated=False)
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64,)
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128,)
        
        '''
        print("RESNET15_CC")
        net = Conv2D( filters=32, kernel_size=7, activation='elu', padding='same', data_format="channels_first")(net)
        
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32,  preactivated=False)
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32,)
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32,)
        
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64,  preactivated=False)
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64,)
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64,)
        
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128,  preactivated=False)
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128,)
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128,)
        


        
        
        #pool_size = net.output_shape[-1]
        net= AveragePooling2D(pool_size=(2,2), data_format='channels_first')(net)
        
    
        return net


# In[ ]:




