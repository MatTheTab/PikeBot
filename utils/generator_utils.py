import os
import pandas as pd
import tensorflow as tf
from utils.data_utils import * 
import pickle

def save_object(object, file_path):
    '''
    Saves a Python object to a file using pickle.
    
    Parameters:
        object: The Python object to be saved.
        file_path: The file path where the object will be saved.
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)

def load_object(file_path):
    '''
    Loads a Python object from a pickle file.
    
    Parameters:
        file_path: The file path from which the object will be loaded.
        
    Returns:
        The loaded Python object.
    '''
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

class NPYSequence(tf.keras.utils.Sequence):
    '''
    Data generator class used for reading data dynamically from the specified directory for optimal performance.
    '''
    def __init__(self, folder_path, column_names_file, batch_size=32, target_column = "human"):
        '''
        Initialize the NPYSequence object.
        
        Parameters:
            folder_path (str): The path to the folder containing the npy.gz files.
            column_names_file (str): The path to the file containing column names.
            batch_size (int, optional): The size of each batch (default is 32).
            target_column (str, optional): The name of the target column in the data (default is "human").
        '''
        self.folder_path = folder_path
        self.column_names_file = column_names_file
        self.batch_size = batch_size
        self.file_names = os.listdir(folder_path)
        self.file_names = [f for f in self.file_names if f.endswith('.npy.gz')]
        self.total_samples = self.calculate_total_samples()
        self.current_data = None
        self.target_column = target_column
        self.load_next_file()
        
    def calculate_total_samples(self):
        '''
        Calculate the total number of samples in all files.
        '''
        total_samples = 0
        for file_name in self.file_names:
            file_path = os.path.join(self.folder_path, file_name)
            data = read(file_path, self.column_names_file)
            total_samples += len(data)
        return total_samples
    
    def __len__(self):
        '''
        Returns the number of batches in the sequence.
        '''
        return self.total_samples // self.batch_size
    
    
    def __getitem__(self, index):
        '''
        Gets a batch of data at the given index.
        
        Parameters:
            index (int): The index of the batch.
        
        Returns:
            X (ndarray): Input data of shape (batch_size, n_features).
            y (ndarray): Target data of shape (batch_size,).
        '''
        start_index = index * self.batch_size - self.current_index
        end_index = start_index + self.batch_size
        if self.current_data is None:
            return None, None
        batch_data = self.current_data.iloc[start_index:end_index]
 
        if len(batch_data) < self.batch_size:
            
            self.load_next_file()
            start_index = 0
            end_index = self.batch_size-len(batch_data)
            
            if self.current_data is not None:
                batch_data = pd.concat([batch_data, self.current_data.iloc[start_index:end_index]], axis=0)
                self.current_index = index * self.batch_size + len(batch_data)
        X = batch_data.drop(columns = [self.target_column])
        X = X.values
        y = batch_data[self.target_column]
        y = y.values
        return X, y
    
    def load_next_file(self):
        '''
        Load the next file in the sequence.
        '''
        if not self.file_names:
            self.current_data = None
            return
        file_name = self.file_names.pop(0)
        file_path = os.path.join(self.folder_path, file_name)
        self.current_index=0
        self.current_data = read(file_path, self.column_names_file)