import tensorflow as tf

# loads the MNIST dataset http://yann.lecun.com/exdb/mnist/
mnist = tf.keras.datasets.mnist

# converts the samples from integers to float
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test /255.0

# builds Sequential model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activiation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# each example returns logits 
predictions = model(x_train[:1]).numpy()
predictions

# converts logits to probabilities
tf.nn.softmax(predictions).numpy()

# takes logits and True index to return a scalar loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1],predictions).numpy()

model.compile(ompimizer='adam',
              loss=loss_fn,
            metrics=['accuracy'])

# adjusts model parameters to minimize the loss
model.fit(x_train, y_train, epochs=5)

# checks models performance and will return accuracy on the dataset
model.evaluate(x_test, y_test, verbose=2)

# to return a probability, attach to the softmax:
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])