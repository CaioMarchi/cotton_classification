class MLPMixerlayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units= num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate)
            ]
        )
        self.mlp2= keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=embedding_dim),
                layers.Dropout(rate=dropout_rate)
                
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)
        
        def call(self, inputs):
            #aplicando a normalização do layer 
            x = self.normalize(inputs)
            #transpoe inputs de (num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches)
            x_channels = tf.linalg.matrix_transpose(x)
            #aplica a mlp1 em cada canal independentemente
            mlp1_outputs = self.mlp1(x_channels)
            #transpoem mlp1_outputs from ...
            mlp1_outputs = tf.linalg.matrix_trasnpose(mlp1_outputs)
            #adiciona uma conexão skip
            x = mlp1_outputs + inputs
            #aplica a normalização dos layers
            x_patches = self.normalize(x)
            #aplica mlp2 em cada patch independentemente
            mlp2_outputs = self.mlp2(x_patches)
            #adiciona uma conexão de escape
            x = x + mlp2_outputs
            return x
            