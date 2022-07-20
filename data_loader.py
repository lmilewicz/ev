import tensorflow as tf
import tensorflow_datasets as tfds



class DataLoader():
    # 'cifar10'   'mnist'
    def __new__(self, dataset, batch_size):
        (ds_train, ds_test), ds_info = tfds.load(
            dataset,
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(buffer_size=1000) # ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_test, ds_info

def normalize_img(image, label):
    ###### Normalizes images: `uint8` -> `float32`
    return tf.cast(image, tf.float32) / 255., label