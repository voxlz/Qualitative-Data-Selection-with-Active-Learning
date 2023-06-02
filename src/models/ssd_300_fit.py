
import tensorflow as tf

''' Attempt at rewriting the model to use the functional API. Unfinished. '''


image_input = tf.keras.Input(shape=(300, 300, 3), name="img_input")

# ...

# Model call implementation

class_output = tf.layers.Dense(5, name="score_output")(image_input)
offset_output = tf.layers.Dense(4, name="offset_output")(image_input)

# ...

model = tf.keras.Model(
    inputs=[image_input], outputs=[class_output, offset_output]
)

# Configure the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": tf.keras.losses.MeanSquaredError(),
        "offset_output": tf.keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.MeanAbsoluteError(),
        ],
        "offset_output": [tf.keras.metrics.CategoricalAccuracy()],
    },
    loss_weights={"score_output": 2.0, "offset_output": 1.0},
)

# Train the model
model.fit(
    {"img_input": img_data},
    {"score_output": score_targets, "offset_output": offset_targets},
    class_weight=class_weight, # ajust for classes that show up more or less in the dataset, or are more important
    sample_weight=sample_weight, # give more importance to samples that are underrepresented in ds
    batch_size=32,
    epochs=1,
)

# ALTERNATIVE
train_dataset = tf.data.Dataset.from_tensor_slices(
    ({"img_input": img_data},
     {"score_output": score_targets, "offset_output": offset_targets}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model.fit(
    train_dataset, 
    class_weight=class_weight, # ajust for classes that show up more or less in the dataset, or are more important
    sample_weight=sample_weight, # give more importance to samples that are underrepresented in ds
    epochs=1
)