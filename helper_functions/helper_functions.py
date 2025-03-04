import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd
import tensorflow as tf
import os


def plot_decision_boundary(model, X, y):
  """
  Plot decision_boundary for binary classification model
  """
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                      np.linspace(y_min, y_max, 100))

  x_in = np.c_[xx.ravel(), yy.ravel()]
  y_pred = model.predict(x_in)

  if model.output_shape[-1] > 1:
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)

  else:
    y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.lim(), xx.max())
  plt.ylim(yy.min(), yy.max())



#-----------------------------------------------------------------------------------------------


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
  """
  Plot confusion matrix for classification problem(binary or multi)
  """
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  ax.set(title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)


  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

    plt.text(j, i, f"{cm[i, j]}({cm_norm[i, j]*100:1.f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)



#---------------------------------------------------------------------------


def plot_loss_curves(history):
  """
  Plot loss curves for model.
  loss & val_loss,
  accuracys & val_accuracy.
  Need to assign validation_set when fitting the model.
  """

  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend();



#--------------------------------------------------------------------------


def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """
  img = tf.io.read_file(filename)

  img = tf.image.decode_image(img, channels=3)

  img = tf.image.resize(img, size = [img_shape, img_shape])

  img = img/255.
  return img


#---------------------------------------------------------------------


def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the
  title.
  """
  img = load_and_prep_image(filename)
  pred = model.predict(tf.expand_dims(img, axis=0))

  if len(pred[0]) > 1:
    pred_class = class_names[pred.argmax()]

  else:
    pred_class = class_names[int(tf.round(pred)[0][0])]

  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)



#-----------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  """
  Setup target directory(we'll view images from here)
  Example usage: ("pizza_steak/train/" contains two class "pizza" & "steak")
  If target_dir="pizza_steak/train/"
  target_class="steak"
  It shows view, picked random image of steak class.
  """

  target_folder = target_dir+target_class
  random_image = random.sample(os.listdir(target_folder), 1)

  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")
  print(f"Image shape: {img.shape}")
  return img


#--------------------------------------------------------------------------------------------------


import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  create a tensorboard callback
  dir_name(str): Write a dir_name to save
  experiment_name(str): Write an experiment_name it shows under dir_name
  when open the dir_name folder.

  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
def create_model(model_url, num_classes, image_shape):
  """
  Takes a TensorFlow Hub URL and creates a Keras Sequential model
  with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
    should be equal to number of target classes(for binary 1 neuron).

  Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extraxtor layer and Dense output layer with num_classes outputs.

    Warning: in KerasLayer(trainable=False) !!!
  """
  image_shape = image_shape
  feature_extractor_layer = hub.KerasLayer(model_url,
                                          trainable=False,
                                          input_shape=image_shape+(3,),
                                          name="feature_extractor_layer")

  model = keras.Sequential([
    feature_extractor_layer,
    keras.layers.Dense(num_classes, activation="softmax",
                         name="output_layer")
    
  ])

  return model


#----------------------------------------------------------------------------------------------


import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """

  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()


#--------------------------------------------------------------------------------


import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path(str): target directory

  Returns:
   A print out of:
     number of subdirectories in dir_path
     number of images (files) in each subdirectory
     name of each subdirectory
  """

  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


#------------------------------------------------------------------------------------------------------







