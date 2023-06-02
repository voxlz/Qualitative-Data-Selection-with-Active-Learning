import tensorflow as tf
# import tensorflow_addons as tfa
from models.ssd_300 import Sequential

from models.vgg16 import VGG16
layers = tf.keras.layers

class Oracle(tf.keras.Model):
    ''' Oracle model for active learning. Takes a model, puts an additional dense layer on top and trains it to predict the loss of unseen images.'''
    def __init__(self, trained_model: tf.keras.Model):
        super().__init__()

        self.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=[
                # tfa.metrics.RSquare(name="R2"), 
                tf.metrics.MeanAbsoluteError(name="MAE"), 
                tf.metrics.MeanAbsolutePercentageError(name="MAPE"), 
                tf.metrics.MeanSquaredError(name="MSE")]
        )

        # Create new model
        # self.vgg16 = VGG16(
        #     include_top=True,
        #     weights=None, # will be overwritten by set_weights
        #     input_shape=(224, 224, 3),
        #     classifier_activation="softmax",
        # )
        # self.vgg16.set_weights(trained_model.get_weights())
        
        trained_model.summerize()

        self.vgg_output = Sequential(self.vgg16.layers[-3:])
        self.lay_neg_3_output = Sequential(self.vgg16.layers[-5:-3])
        self.lay_neg_5_output = Sequential(self.vgg16.layers[-7:-5])
        self.lay_neg_7_output = Sequential(self.vgg16.layers[-9:-7])
        self.lay_neg_9_output = Sequential(self.vgg16.layers[-11:-9])
        self.lay_neg_11_output = Sequential(self.vgg16.layers[-13:-11])
        self.lay_neg_13_output = Sequential(self.vgg16.layers[-15:-13])
        self.lay_neg_15_output = Sequential(self.vgg16.layers[-17:-15])
        self.lay_neg_17_output = Sequential(self.vgg16.layers[:-17])
        
        # outputs = [layer.output for layer in self.vgg16.layers] 
        
        if (True):
            test_tensor = tf.random.uniform(shape=[32, 224, 224, 3])
            a1 = self.vgg16(test_tensor)
            x_8 = self.lay_neg_17_output(test_tensor)
            x_7 = self.lay_neg_15_output(x_8)
            x_6 = self.lay_neg_13_output(x_7)
            x_5 = self.lay_neg_11_output(x_6)
            x_4 = self.lay_neg_9_output(x_5)
            x_3 = self.lay_neg_7_output(x_4)
            x_2 = self.lay_neg_5_output(x_3)
            x_1 = self.lay_neg_3_output(x_2)
            x   = self.vgg_output(x_1)
            tf.assert_equal(a1, x)
        
        self.concat   = layers.Concatenate()
        self.flatten  = layers.Flatten()
        self.dense_1  = layers.Dense(20, activation='relu')
        # self.dense_2  = layers.Dense(100, activation='relu')
        self.dense_2  = layers.Dense(1, activation='sigmoid')
        self.vgg16.trainable = False
        
    def call(self, inputs):
        # if x is tuple: x = self.concat(x)
        
        # output = self.vgg16(inputs)
        
        x_8 = self.lay_neg_17_output(inputs)
        x_7 = self.lay_neg_15_output(x_8)
        x_6 = self.lay_neg_13_output(x_7)
        x_5 = self.lay_neg_11_output(x_6)
        x_4 = self.lay_neg_9_output(x_5)
        x_3 = self.lay_neg_7_output(x_4)
        x_2 = self.lay_neg_5_output(x_3)
        x_1 = self.lay_neg_3_output(x_2)
        x   = self.vgg_output(x_1)
        
        # Flatten
        x_1 = self.flatten(x_1)
        x_2 = self.flatten(x_2)
        x_3 = self.flatten(x_3)
        x_4 = self.flatten(x_4)
        x_5 = self.flatten(x_5)
        x_6 = self.flatten(x_6)
        x_7 = self.flatten(x_7)
        x_8 = self.flatten(x_8)
        
        x = self.concat([x, x_1, x_3, x_5, x_7, x_8])
        # x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        # x = self.dense_3(x)
        return tf.squeeze(x)
    
   