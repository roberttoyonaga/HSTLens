
# coding: utf-8

# In[1]:


from keras.layers import Activation, Conv2D, Add, MaxPooling2D, LeakyReLU

from keras.layers.normalization import BatchNormalization

def pre_activated_resnet_block(data_in, n_filters_in, n_filters_out, downsampling=False, preactivated=False, kernel=3, pool= True):
    print ("pre- resnet block entered!")
    print('resnet16 blocks')
    print('kernel= '+str(kernel))
   
    net = Conv2D( filters=n_filters_in, kernel_size=kernel, activation=None, padding='same',
                 input_shape=(data_in.shape[0],data_in.shape[1], data_in.shape[2]), data_format="channels_first")(data_in)
    net = BatchNormalization(axis=1)(net) #axis is set to the dimension which hold the colour channels
    net =LeakyReLU()(net) #Activation('relu')(net)

    net = Conv2D( filters=n_filters_in, kernel_size=kernel, activation=None, padding='same', data_format="channels_first")(net)
    net = BatchNormalization(axis=1)(net)
    net =  LeakyReLU()(net)#Activation('relu')(net)

    net = Conv2D( filters=n_filters_out, kernel_size=kernel, activation=None, padding='same', data_format="channels_first")(net)
    net = BatchNormalization(axis=1)(net)
    net = LeakyReLU()(net) #Activation('relu')(net)
    
    if pool ==True:
        net= MaxPooling2D(pool_size=(2,2), data_format='channels_first')(net)
        #net=Dropout(0.25)(net)
    
    
    # Shortcut branch
    if n_filters_in != n_filters_out:
        #to increase depth
        shortcut = Conv2D( filters=n_filters_out, kernel_size=1, activation=None, padding='same',
                     input_shape=(data_in.shape[0],data_in.shape[1], data_in.shape[2]), data_format="channels_first")(data_in)

        if pool==True:
            shortcut= MaxPooling2D(pool_size=(2,2), data_format='channels_first')(shortcut)
        shortcut = BatchNormalization(axis=1)(shortcut)

    elif pool==True:
        shortcut = MaxPooling2D(pool_size=(2,2), data_format='channels_first')(data_in)
        shortcut = BatchNormalization(axis=1)(shortcut)
    else:
        shortcut = data_in
        shortcut = BatchNormalization(axis=1)(shortcut)
    
    print("shorcut "+str(shortcut.shape)+ " net "+ str(net.shape))
    output = Add()([net, shortcut])

    return output





