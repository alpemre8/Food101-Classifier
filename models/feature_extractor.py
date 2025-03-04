import tensorflow as tf
import tf_keras as keras

def create_feature_extractor():

  base_model = keras.applications.EfficientNetB0(include_top=False)
  base_model.trainable = False

  inputs = keras.layers.Inpus(shape=(224, 224, 3), name="input_layer")
  x = base_model(inputs, training=False)
  x = keras.layers.GlobalAveragePooling2D(name="pooling_layer_2d")(x)
  x = keras.layers.Dense(units=101)(x)
  outputs = keras.layers.Activation("softmax", dtype_tf.float32, name="softmax_float32")(x)

  model = keras.Model(inputs, outputs)

  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
               optimizer=keras.optimizers.Adam(),
               metrics=["accuracy"])

# Create feature extractor model, that model's name is model
