from keras_tuner import HyperModel

class BuildHypermodel(HyperModel):
    def build(self,hp):
        import keras_tuner
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import regularizers
        
        
        #criando um objeto do           modelo
        model = keras.Sequential([
        #adicionando a primeira cama convoluscional (1)
        keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [2,5]),
        activation='relu',
        input_shape=(600, 600, 3)),
        keras.layers.MaxPooling2D((2,2)),
                    
        #adicionando a segunda camda convolucional (2) 
        keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [2,5]),
        activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
                    
        #adicionando um camada flatten
        keras.layers.Flatten(),
                    
        #adicionando uma camada densa (1)
        #
        keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu',
        kernel_regularizer= regularizers.l2(0.001)),
        #adicionando segunda camada densa (2)
        keras.layers.Dense(
        units= hp.Int('dense_2_units', min_value=16, max_value=64, step=16),
        activation='relu'),
        #output
        keras.layers.Dense(5,
        activation=hp.Choice('dense_3_activation',
                            ['relu', 'sigmoid']))
                    ])
                    
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
                    
        #compilando o modelo
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
                    
        return model
