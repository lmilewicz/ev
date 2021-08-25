import tensorflow as tf

class DataLoader():
    def __new__(self, dataset):
        if dataset == 'MNIST':
            dataset = tf.keras.datasets.mnist.load_data()
            (train_images, train_labels), (test_images, test_labels) = dataset
            train_images, test_images = train_images / 255.0, test_images / 255.0
        else:
            raise NotImplementedError("Dataset {} for DataLoader not implemented".format(dataset))
        
        return (train_images, train_labels), (test_images, test_labels)
