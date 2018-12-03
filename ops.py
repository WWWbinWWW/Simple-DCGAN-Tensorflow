import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

######################################
#Layer
######################################
def conv(x, channel, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
	with tf.variable_scope(scope):
		if pad_type == 'zero':
			x = tf.pad(x, [[0,0], [pad,pad], [pad,pad], [0,0]])
		if pad_type == 'reflect':
			x = tf.pad(x, [[0,0], [pad,pad], [pad,pad], [0,0]], mode='REFLECT')

		x = tf.layers.conv2d(inputs=x, filters=channel,
							kernel_size=kernel, kernel_initializer=weight_init,
							kernel_regularizer=weight_regularizer,
							use_bias=use_bias, strides=stride)

		return x

def dconv(x, channel, kernel=4, stride=2, use_bias=True, scope='dconv_0'):
	with tf.variable_scope(scope):
		x = tf.layers.conv2d_transpose(inputs=x, filters=channel,
										kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
										strides=stride, padding='SAME', use_bias=use_bias)

		return x

def flatten(x):
	return tf.layers.flatten(x)

def linear(input_, m_shape, output_size, scope=None, stddev=0.2):
	with tf.variable_scope(scope or "Linear"):
		try:
			matrix = tf.get_variable("Matrix", [m_shape, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
		except ValueError as err:
			msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
			err.args = err.args + (msg,)
			raise
		bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(0.0))
		return tf.matmul(input_, matrix) + bias

######################################
#ResBLock
######################################
def resblock(x_init, channel, use_bias=True, scope='resblock'):
	with tf.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = conv(x_init, channel, kernel=3, stride=1, pad=1, use_bias=use_bias)
			x = instance_norm(x)
			x = relu(x)
		with tf.variable_scope('res2'):
			x = conv(x, channel, kernel=3, stride=1, pad=1, use_bias=use_bias)
			x = instance_norm(x)

		return x+x_init

######################################
#Activation function
######################################
def lrelu(x, alpha=0.2):
	return tf.nn.leaky_relu(x, alpha)

def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.nn.tanh(x)

######################################
#Norm
######################################
def instance_norm(x, scope='instance_norm'):
	return tf_contrib.layers.instance_norm(x,
											epsilon=1e-5,
											center=True, scale=True,
											scope=scope)

def batch_norm(x, scope='batch_norm'):
	return tf_contrib.layers.batch_norm(x,
										decay=0.9,
										epsilon=1e-5,
										scale=True,
										is_training=True,
										scope=scope)

######################################
#Loss function
######################################
def discriminator_loss(real, fake):
	#使用普通Loss
	real_loss = 0
	fake_loss = 0

	#原始Loss
	#real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
	#fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

	#WGAN_loss
	real_loss = -tf.reduce_mean(real)
	fake_loss = tf.reduce_mean(fake)

	loss = real_loss + fake_loss

	return loss

def generator_loss(fake):
	#使用普通Loss
	fake_loss = 0

	#原始Loss
	#fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

	fake_loss = -tf.reduce_mean(fake)

	loss = fake_loss

	return fake_loss

