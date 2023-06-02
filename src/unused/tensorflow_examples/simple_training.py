import tensorflow as tf

cifar100 = tf.keras.datasets.cifar100

# Load dataset of images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
input_shape=(50000, 32, 32, 3)

# Preprocessing (without 2%, with 30%)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model with layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=input_shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=input_shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
]) # evaluation accuracy: 0.2908

# loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compile model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
model.evaluate(x_test, y_test, verbose=2)

## -----------
# get an output vector (called logits)
logits = model(x_train[:1]).numpy()

# turn output into "probabilities" (or something close to it)
predictions = tf.nn.softmax(logits).numpy()

# ====== OR =======

# Wrap model to return a probability instead of classification
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Get probability
predictions = probability_model(x_train[:1]).numpy()
## -----------

# Get loss for your x prediction
loss = loss_fn(y_train[:1], predictions).numpy()
print(logits, predictions, loss)

print("TensorFlow version:", tf.__version__)