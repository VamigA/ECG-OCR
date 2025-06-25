import numpy as np
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

with tf.device('/GPU:0'):
	part1 = tf.keras.Sequential([
		tf.keras.layers.Input((2048,)),
		*[tf.keras.layers.Dense(8192, activation='sigmoid') for _ in range(5)],
	])

with tf.device('/GPU:1'):
	part2 = tf.keras.Sequential([
		tf.keras.layers.Input((8192,)),
		*[tf.keras.layers.Dense(8192, activation='sigmoid') for _ in range(5)],
		tf.keras.layers.Dense(10),
	])

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x_batch, y_batch):
	with tf.GradientTape() as tape:
		with tf.device('/GPU:0'):
			x1 = part1(x_batch, training=True)
		with tf.device('/GPU:1'):
			y_pred = part2(x1, training=True)
			loss = loss_fn(y_batch, y_pred)

	variables = part1.trainable_variables + part2.trainable_variables
	grads = tape.gradient(loss, variables)
	optimizer.apply_gradients(zip(grads, variables))

	return loss

x_data = tf.convert_to_tensor(np.random.rand(256, 2048).astype(np.float32))
y_data = tf.convert_to_tensor(np.random.rand(256, 10).astype(np.float32))

batch_size = 128
epochs = 20
steps_per_epoch = x_data.shape[0] // batch_size

for epoch in range(epochs):
	for step in range(steps_per_epoch):
		x_batch = x_data[step * batch_size:(step + 1) * batch_size]
		y_batch = y_data[step * batch_size:(step + 1) * batch_size]
		loss = train_step(x_batch, y_batch)

	print(f'Epoch {epoch + 1}/{epochs} completed, loss: {loss}!')