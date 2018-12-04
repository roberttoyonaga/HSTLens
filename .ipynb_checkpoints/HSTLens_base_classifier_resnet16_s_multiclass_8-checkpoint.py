
# coding: utf-8



#import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping
from keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt

#need these
import time
from sklearn.metrics import roc_curve
import math


class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'epoch_weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1

class BaseKerasClassifier():
    def __init__(self, pos_weight=1,
                       n_epochs=100,
                       batch_size=32,
                       learning_rate=0.001,
                       learning_rate_drop=0.1,
                       learning_rate_steps=3,
                       output_nbatch=100,
                       val_nepoch=5):
        """
        Initialisation
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.output_nbatch = output_nbatch
        self.pos_weight = pos_weight # what ratio to weight the positive samples (lenses)
        self.val_nepoch = val_nepoch
        self.learning_rate_drop = learning_rate_drop
        self.learning_rate_steps = learning_rate_steps
        self.built=False
        self.fitted=False
        self.model =None

    def _model_definition(self, net):
        """
        Function which defines the model from the provided source layer
        to the output of the DenseLayer, before the dense layer !
        """
        return net
        
        
    def _build(self):
        print ("building 7 ")
        n_x = int(100) #height/width
        n_c = int(1) # (n_samples,n_bands, width, height)

        #self.l_x= Input(shape=(101, 101,4))
        inputs= Input(shape=(n_c, n_x, n_x))
        
       
        net = self._model_definition(inputs) 

        out=Flatten()(net) #may need to remove this later, but always needed before fully connected layers 
        
#         out = Dense(512, activation='relu')(out) 
#         out=Dropout(0.30)(out) 
        out = Dense(128, activation='relu')(out) 
        out=Dropout(0.40)(out)
        out = Dense(32, activation='relu')(out) 
        out=Dropout(0.25)(out)
#         out = Dense(64, activation='relu')(out) 
#         out=Dropout(0.30)(out) 
        predictions = Dense(4, activation='softmax')(out)
        #predictions= Activation('relu')(predictions) 

        #compile the model accounting for all computaitons form inputs to outputs
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        self.built=True
        

    def _fit(self,data,labels):
    #def _fit(self,data,labels,val):
        '''
        Accepts input tensors
        Returns nothing
        Performs basic data augmentation, calls back to learning rate scheduler
        
        '''

        print("...Fitting model ...")
        #build if not built
        if self.built is False:
            self._build(data, labels)
        
       
        lrate = LearningRateScheduler(self.step_decay)
        
        self.batch_history = LossHistory()
#         callbacks_list = [lrate, self.batch_history, EarlyStopping(monitor='val_loss', patience=30), WeightsSaver(5)]
        callbacks_list = [lrate, self.batch_history, EarlyStopping(monitor='val_loss', patience=30)]

        
        self.model.fit(data, labels, batch_size=self.batch_size, callbacks=callbacks_list,
                       shuffle= True, validation_split = 0.5, epochs= self.n_epochs)
        
#         self.model.fit(data, labels, batch_size=self.batch_size, callbacks=callbacks_list,
#                        shuffle= False, validation_split = 0,validation_data =val, epochs= self.n_epochs)        
        self.fitted=True
        
        
    def step_decay(self, epoch):
        '''
        Accepts an epoch number
        Returns a learning rate
        This function will decrease the learning rate in steps by a constant factor
        '''
        initial_lrate = self.learning_rate
        drop = 0.5
        epochs_drop = 15 #every x epochs drop the learning rate
        lrate = self.learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        print ("step decay entered. Learning rate: "+str(lrate))
        return lrate


    def _save(self, filename):
        '''
        Accepts filename string
        Returns nothing, but saves weights as hdp5 file
        '''
        print("saving to " + filename + "...")
        self.model.save(filename)
        
        
    def _load(self, filename ):
        '''
        Accepts filename and input tensors
        Returns nothing. Initialize a model object and load the weights form specified file
        '''
   
        self.model=load_model(filename)
        self.fitted=True
        print("Model loaded")
        
        
    def _predict(self, X, y= None, discrete = True):
        """
        Accepts feature tensor
        Returns class predictions for X or probabilities depending on whether "discrete" is set to true
        """
        print("...Generating Predictions ...")
#         if (y !=None) and ( self.fitted is False):
#             self._fit(X, y)
            
        try:    
            predictions = self.model.predict(X)

            #round to 1 or 0. Could also use: np.where(predictions > threshold, upper, lower)
            if discrete==True: 
                rounded = [round(x[0]) for x in predictions] 
                return rounded
            else:
                return predictions
        
        except:            
            print("you need to fit the model first")
            return None
    
    def eval_purity_completeness(self, X, y):
        """
        Accepts input tensors
        Returns purity and completeness calculations
        Purity = N(true positive) / [N(true positive) + N(false positive)]
        Compl. = N(true positive) / [N(true positive) + N(false negative)]
        """
#         if self.fitted is False:
#             self._fit(X, y)
            
        predictions = self._predict(X)
        
        n_fp=0.
        n_tp=0.
        n_fn=0.
        for prediction in range(len(predictions)):
            if predictions[prediction]==1 and y[prediction]==1:
                n_tp+=1.
            elif predictions[prediction]==1 and y[prediction]==0:
                n_fp +=1.
            elif predictions[prediction]==0 and y[prediction]==1:
                n_fn+=1.

        pur = n_tp / ( n_tp + n_fp )
        comp= n_tp / ( n_tp + n_fn )
        

        return pur, comp #, pur1, comp1
    
    
    def eval_ROC(self, X, y):
        '''
        Accepts input tensors
        Returns fals positive rate and true positive rate a s 1d arrays. Returns thresholds too
        '''
        
#         if self.fitted is False:
#             self._fit(X, y)
        
        y_pred_keras = self.model.predict(X).ravel()
        fpr, tpr, t = roc_curve(y, y_pred_keras)

        return fpr, tpr, t
      
class LossHistory(Callback):
    
    
    def __init__(self):
        self.losses = []
        self.epoch_losses= []
        
    
    def on_epoch_begin(self, epoch, logs=None):
        self.losses = []
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_acc'))
        
    def on_epoch_end(self, epoch, logs=None):
        
        self.epoch_losses.append(self.losses)
        #self.epoch_losses.append(logs.get('val_acc'))
    

            
        