import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Mnist data set: 70,000 handwritten number images
mnist = tf.keras.datasets.mnist

# x_train = a set of 60,000 handwritten images (size of 28*28) 
# x_test = a set of 10,000 handwritten images (size of 28*28)
# y_train = a set of 60,000 label of image (0~9)
# y_test = a set of 10,000 label of image (0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# DataPreprocess: turn pixel values(0~255.0) into values(0~1.0)
x_train, x_test = x_train / 255.0, x_test / 255.0

# model: Flatten-Dense-dropout-Dense
# Flatten: 28x28 --> 1 dimension array
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

# Compile the model
# Loss function: adam(Adaptive Momentum estimation)
# Metrics: Accuracy(정확도)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Training Model(training data, label, epochs)
# Epoch: How many times to train the whole training data
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the model
model.save('model1.h10')

# Evaluate
# evlauate(test data, label)
test_acc = model.evaluate(x_test,  y_test, verbose=2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1])
plt.legend(loc='lower right')
plt.show()

print(test_acc)

plt.savefig(sys.stdout.buffer)
sys.stdout.flush()