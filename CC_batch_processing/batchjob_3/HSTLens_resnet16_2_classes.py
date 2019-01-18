
# coding: utf-8

# In[2]:



from HSTLens_base_classifier_resnet16_2_classes import BaseKerasClassifier
from HSTLens_blocks_resnet16_2_classes import pre_activated_resnet_block


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
        print("multiclass_7")
        
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32, kernel=5) #100x100
        net = pre_activated_resnet_block(net, n_filters_in=16, n_filters_out=32, kernel=3,pool=True) #50x50
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64, kernel=3,pool=False) #25x25
        net = pre_activated_resnet_block(net, n_filters_in=32, n_filters_out=64, kernel=3, pool=True) #25x25
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128, kernel=3,pool=False) #12x12
        net = pre_activated_resnet_block(net, n_filters_in=64, n_filters_out=128, kernel=3, pool=True) #12x12

        return net







