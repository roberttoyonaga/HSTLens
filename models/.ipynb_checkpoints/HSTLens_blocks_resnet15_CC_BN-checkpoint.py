
# coding: utf-8

# In[1]:


from keras.layers import Activation, Conv2D, ELU, Add, MaxPooling2D, PReLU

from keras.layers.normalization import BatchNormalization

def pre_activated_resnet_block(data_in, n_filters_in, n_filters_out, downsampling=False, preactivated=False):
    print ("pre- resnet block entered!")
    print("RESNET15_C BLOCKS_2")
   
    net = Conv2D( filters=n_filters_in, kernel_size=3, activation=None, padding='same',
                 input_shape=(data_in.shape[0],data_in.shape[1], data_in.shape[2]), data_format="channels_first")(data_in)
    net = BatchNormalization(axis=1)(net) #axis is set to the dimension which hold the colour channels
    net = Activation('elu')(net)

    net = Conv2D( filters=n_filters_in, kernel_size=3, activation=None, padding='same', data_format="channels_first")(net)
    net = BatchNormalization(axis=1)(net)
    net = Activation('elu')(net)

    net = Conv2D( filters=n_filters_out, kernel_size=3, activation=None, padding='same', data_format="channels_first")(net)
    net = BatchNormalization(axis=1)(net)
    net = Activation('elu')(net)
    
    if downsampling==True:
        net= MaxPooling2D(pool_size=(2,2), data_format='channels_first')(net)
    #net=Dropout(0.25)(net)
    
    # Shortcut branch
    if n_filters_in != n_filters_out:
        #to increase depth
        shortcut = Conv2D( filters=n_filters_out, kernel_size=1, activation=None, padding='same',
                     input_shape=(data_in.shape[0],data_in.shape[1], data_in.shape[2]), data_format="channels_first")(data_in)
        if downsampling==True:
            shortcut= MaxPooling2D(pool_size=(2,2), data_format='channels_first')(shortcut)
        shortcut = BatchNormalization(axis=1)(shortcut)

    else:
        #depth does not need to be increased
        if downsampling==True:
            shortcut= MaxPooling2D(pool_size=(2,2), data_format='channels_first')(shortcut)
        shortcut = BatchNormalization(axis=1)(shortcut)
    
    print("shorcut "+str(shortcut.shape)+ " net "+ str(net.shape))
    output = Add()([net, shortcut])
    output = Activation('elu')(output)

    return output





