import tensorflow as tf
import tensorflow_datasets as tfds


def load_data():
  (train_data, test_data), ds_info = tfds.load(name="food101",
                                              split=["train", "validation"],
                                              shuffle_files=True,
                                              as_supervised=True,
                                              with_info=True)

  
  def preprocess_img(image, label):

    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32), label


  train_data = train_data.map(map_funch=preprocess_img,
                             num_parallel_calls=tf.data.AUTOTUNE)
  train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
  test_data = test_data.map(map_func=preprocess_img,
                           num_parallel_calls=tf.data.AUTOTUNE)
  test_data = test_data.batch(batch_size=32).prefetch(tf.data.AUTOTUNE)
  return train_data, test_data, ds_info


  

