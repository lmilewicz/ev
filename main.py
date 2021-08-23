# Plan:

import tensorflow as tf
import tensorflow_probability as tfp

from blueprint import DenseLayer, Blueprint

n_epochs = 10
mnist_dataset = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist_dataset
train_images, test_images = train_images / 255.0, test_images / 255.0


import time
start_time = time.time()

### Normal
blueprint_graph = [[1], [0, 1], [1, 1, 1]]
input_shape = (28, 28, 1)
layerType = DenseLayer
blueprint_object = Blueprint(blueprint_graph, input_shape, layerType)

model = blueprint_object.get_model()

model.fit(x=train_images,
            y=train_labels,
            epochs=n_epochs,
            use_multiprocessing=True,
            batch_size=64)

print('Execution Time: %s' % (time.time()-start_time))

model.summary()

### TFP
# model_tfp = get_model((28, 28, 1), dtype=dtype, layerType=DenseFlipout)

# model_tfp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
# model_tfp.fit(x=train_images,
#             y=train_labels,
#             epochs=10,
#             use_multiprocessing=True,
#             batch_size=64)

# model_tfp.summary()