from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import keras_tuner
import os

class ModelTuner:
    def __init__(self, image_size, path):    
        self.image_size = image_size
        self.path = path
            
    def get_log_dir(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("Pasta criada com sucesso")
        else:
            print("A pasta já existe")
        return self.path
    
    def tuner_bayesian_optimization(self, model, objective, training_Data, valid_Data, max_trials ,epochs):
        tuner = keras_tuner.BayesianOptimization(hypermodel= model,
                                                    objective= objective,
                                                    max_trials= max_trials,
                                                    executions_per_trial= 1,
                                                    directory= self.get_log_dir(),
                                                    project_name= "bayesian_optimization")
            
        print(tuner.search_space_summary())
        tuner.search(training_Data,
                epochs= epochs, 
                validation_data= valid_Data,
                callbacks= [keras.callbacks.TensorBoard(log_dir= self.get_log_dir())])
            
        return tuner

    def get_best_model(self, tuned_model):
        best_hps_bo = tuned_model.get_best_hyperparameters()
            
        return print(best_hps_bo)