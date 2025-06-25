import numpy as np
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

model = tf.keras.Sequential([
	tf.keras.layers.Input((2048,)),
	*[tf.keras.layers.Dense(8192, activation='sigmoid') for _ in range(10)],
	tf.keras.layers.Dense(10),
])

model.compile(optimizer='adam', loss='mse')

x = np.random.rand(10000, 2048).astype(np.float32)
y = np.random.rand(10000, 10).astype(np.float32)

model.fit(x, y, epochs=20, batch_size=64)