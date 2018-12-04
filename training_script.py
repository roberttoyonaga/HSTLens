# 4 classes
import numpy as np

#load numpy binaries
pos = np.load('/home/toyonaga/scratch/HSTLens_datasets/batchjob_3_pos.npy')
neg = np.load('/home/toyonaga/scratch/HSTLens_datasets/batchjob_3_neg.npy')
crowded = np.load('/home/toyonaga/scratch/HSTLens_datasets/batchjob_3_crowded.npy')
multisource = np.load('/home/toyonaga/scratch/HSTLens_datasets/batchjob_3_multisource.npy')

#one-hot-encode
py =np.ones((pos.shape[0],4))
for i in range(py.shape[0]): py[i] =[1,0,0,0]
ny =np.ones((neg.shape[0],4))
for i in range(ny.shape[0]): ny[i] =[0,1,0,0]
cy =np.ones((pos.shape[0],4))
for i in range(cy.shape[0]): cy[i] =[0,0,1,0]
my =np.ones((neg.shape[0],4))
for i in range(my.shape[0]): my[i] =[0,0,0,1]

#concat the data sets into features and labels
x_combined =np.concatenate((pos, neg, crowded, multisource))
y_combined =np.concatenate((py, ny, cy, my)) 

#shuffle the data sets
s3 = np.arange(x_combined.shape[0])
np.random.shuffle(s3)
x_combined_shuf =x_combined[s3]
y_combined_shuf = y_combined[s3]



from HSTLens_resnet16_s_multiclass_6 import deeplens_classifier

my_model = deeplens_classifier(n_epochs=100, batch_size=32)
my_model._build() 

my_model._fit(x_combined_shuf[:50000],y_combined_shuf[:50000])

#my_model.model.save("real_lenses_weights") # issue with json serializing
my_model.model.save_weights("multiclass_ensemble_1")
# my_model.model.load_weights("combined_nonsubtracted_weights_resnet16_s_h4_4")

