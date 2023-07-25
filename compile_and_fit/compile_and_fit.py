import tensorflow_docs as tfdocs
from tensorflow import keras
import tensorflow as tf
import os


class CompilerAndFit:
    def __init__(self, train_Data, valid_Data, path, monitor, optimizer= None):
        self.train_Data = train_Data
        self.valid_Data = valid_Data
        self.optimizer = optimizer
        self.path = path
        self.monitor = monitor
        
    def __get_steps_per_epoch(self,):
        n_train = (len(self.train_Data))
        for image_batch, labels_batch in self.train_Data:
            print("Tamanho dos Batchs", labels_batch.shape)
            break
        steps_per_epoch = n_train//labels_batch.shape[0]
        return steps_per_epoch
        
    def __get_lr_schedule(self,):
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps= self.__get_steps_per_epoch() * 100,
            decay_rate=1,
            staircase=False
            )
        return lr_schedule
    
    def __get_log_dir(self,):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("Pasta criada com sucesso")
        else:
            print("A pasta já existe")
        return self.path

    def __get_callbacks(self,):
        return [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.EarlyStopping(monitor=self.monitor, patience=20, mode="max"),
            tf.keras.callbacks.TensorBoard(log_dir= self.__get_log_dir()),
        ]

    def __get_optimizer(self, optimizer):
        if self.optimizer == None:
            return tf.keras.optimizers.Adam(self.__get_lr_schedule())
        elif self.optmizer == "SGD":
            return tf.keras.optimizers.SGD(self._get_lr_schedule())

    def compile_and_fit(self, model, max_epochs, optimizer= None):
        model.compile(optimizer= self.__get_optimizer(optimizer),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics= [
                                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                                keras.metrics.SparseTopKCategoricalAccuracy(2, name="top5-acc")]
                            )
        model.summary()
        
        history = model.fit(
            self.train_Data,
            steps_per_epoch= self.__get_steps_per_epoch(),
            validation_data= self.valid_Data,
            epochs= max_epochs,
            callbacks= self.__get_callbacks()
        )
        return history
    
    def plot_acc(self, history):
        #plotando o gráfico de acurácia
        plotter = tfdocs.plots.HistoryPlotter(metric = 'acc')
        history_dict = {'history': history}
        plotter.plot(history_dict)
        
        