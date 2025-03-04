import tensorflow as tf
import tf_keras as keras
import os

def fine_tuned_model(model, train_data, test_data, epochs=10, learning_rate=1e-4):
  """
  Model parameter is feature extractor model
  """

  early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=3)
  checkpoint_path = os.path.expanduser("~/fine_tune_checkpoints")
  os.makedirs(checkpoint_path, exist_ok=True)
  checkpoint_file = os.path.join(checkpoint_path, "best_model.keras")

  model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_file,
                                                    save_best_only=True,
                                                    monitor="val_loss")

  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                               factor=0.2,
                                               patience=2,
                                               verbose=1,
                                               min_lr=1e-7)
  
  
  model.trainable = True

  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
               optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
               metrics=["accuracy"])


  history = model.fit(train_data,
                     epochs=epochs,
                     steps_per_epoch=len(train_data),
                     validation_data=test_data,
                     validation_steps=int(0.15 * len(test_data)),
                     callbacks=[model_checkpoint,
                               early_stopping,
                               reduce_lr])
  return model, history
  

