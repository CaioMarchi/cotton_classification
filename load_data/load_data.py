from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import os

class LoadData:
    def __init__(self, path, num_batches, image_size):
        self.path = path
        self.num_batches = num_batches
        self.image_size = image_size
        
    def __get_class_names(self,):
        files = os.listdir(self.path)
        files = [file for file in files
                if file != ".DS_Store"]
        return files
    
    def get_data(self,):
        data = image_dataset_from_directory(self.path, batch_size = self.num_batches, 
                                            class_names= self.__get_class_names(),
                                            image_size= (self.image_size, self.image_size))
        
        AUTOTUNE = tf.data.AUTOTUNE
        data = data.cache().prefetch(buffer_size = AUTOTUNE)
        return data
        