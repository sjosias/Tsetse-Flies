from torch.utils import data
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, image_transform = None ):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.image_transform = image_transform
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        '''
        X - image
        y - list of coords
        '''
        
        ID = self.list_IDs[index]
        X, y = self.__data_generation(ID)

        return X, y

    def __data_generation(self, ID):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
     
        # Read in image
        temp_img = io.imread("../"+ID)

        X = temp_img 
        y = []

        # Store class
        labels = int(self.labels[ID])
        
        X = Image.fromarray(X, mode = 'RGB') # This is important when working with colour images
        if self.image_transform is not None:
            X = self.image_transform(X)

        return X, labels