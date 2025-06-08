import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
	tf.keras.layers.Input((2048,)),
	tf.keras.layers.Dense(2048, activation='relu'),
	tf.keras.layers.Dense(2048, activation='relu'),
	tf.keras.layers.Dense(10),
])

model.compile(optimizer='adam', loss='mse')

x = np.random.rand(10000, 2048).astype(np.float32)
y = np.random.rand(10000, 10).astype(np.float32)

model.fit(x, y, epochs=20, batch_size=64)