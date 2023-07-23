import tensorflow_docs as tfdocs
from tensorflow import keras
import tensorflow as tf
import os


class CompilerAndFit:
    def __init__(self, train_data,optmizer, path, monitor):
        self.train_data = train_data
        self.num_batches
        self.optmizer = optmizer
        self.path = path
        self.monitor = monitor
        
    def __steps_per_epoch(self):
        n_train = (len(self.train_data))
        
    def __get_lr_schedule(self):
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps= steps_per_epoch*100,
            decay_rate=1,
            staircase=False
            )
        return lr_schedule
    
    def __get_log_dir(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("Pasta criada com sucesso")
        else:
            print("A pasta já existe")
        return self.path

    def __get_callbacks(self):
        return [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.EarlyStopping(monitor=self.monitor, patience=20, mode="max"),
            tf.keras.callbacks.TensorBoard(log_dir= self.__get_log_dir()),
        ]

    def __get_optimizer(self,):
        if self.optmizer == None:
            return tf.keras.optimizers.Adam(self.__get_lr_schedule())
        elif self.optmizer == "SGD":
            return tf.keras.optimizers.SGD(self._get_lr_schedule())

    def compile_and_fit(self, model, train_Data, validation_Data, max_epochs, callback_path,optimizer= None):
        if optimizer is None:
            optimizer = self._get_optimizer()
        model.compile(optimizer= optimizer,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics= [
                                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                                keras.metrics.SparseTopKCategoricalAccuracy(2, name="top5-acc")]
                            )
        model.summary()
        
        history = model.fit(
            train_Data,
            steps_per_epoch= steps_per_epoch,
            validation_data= validation_Data,
            epochs= max_epochs,
            callbacks= get_callbacks(callback_path)
        )
        return history