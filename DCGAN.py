from ops import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class DCGAN(object):
	def __init__(self, b_ch=64, e_ch=1024, n_d=4, n_g=5, b_size=4, e_size=128, channel=3,
				b_s=32, z_size=100, beta1=0.5, lr=0.00002, ld=10, epoch=10, iterations=1000, data_num=100,
				data_dir='././img_data_porsche'):
		self.data_dir = data_dir
		self.epoch = epoch
		self.iterations = iterations
		self.data_num = data_num
		self.lr = lr
		self.ld = ld
		self.beta1 = beta1
		self.b_s = b_s
		self.b_ch = b_ch
		self.e_ch = e_ch
		self.channel = channel
		self.n_d = n_d
		self.n_g = n_g
		self.z_size = z_size
		self.b_size = b_size
		self.e_size = e_size

	def discriminator(self, x, reuse=False, scope='discriminator'):
		with tf.variable_scope(scope, reuse=reuse):
			x = conv(x, 64, kernel=4, stride=2, pad=1, use_bias=True, scope='conv0')
			x = lrelu(x)
			x = conv(x, 128, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_1')
			x = lrelu(batch_norm(x, 'dis_norm1'))
			x = conv(x, 256, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_2')
			x = lrelu(batch_norm(x, 'dis_norm2'))
			x = conv(x, 512, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_3')
			x = lrelu(batch_norm(x, 'dis_norm3'))
			x = conv(x, 1024, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_4')
			x = lrelu(batch_norm(x, 'dis_norm4'))

			x = flatten(x)
			y_pred = linear(x, 16384, output_size=1, stddev=0.2)

			return tf.nn.sigmoid(y_pred), y_pred

	def gradient_panalty(self, real, fake, scope="discriminator"):
		alpha = tf.random_uniform(shape=[self.b_s, 1, 1, 1], minval=0., maxval=1.)
		interpolated = alpha * real + (1.-alpha) * fake
		_, logit = self.discriminator(interpolated, reuse=True, scope=scope)

		GP = 0

		grad = tf.gradients(logit, interpolated)[0]
		grad_norm = tf.norm(flatten(grad), axis=1)

		GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

		return GP

	def generator(self, z, reuse=False, scope='generator'):
		with tf.variable_scope(scope, reuse=reuse):
			z = tf.layers.dense(inputs=z, units=16384, activation=None, use_bias=True, name='z')
			z = tf.reshape(z, shape=[-1, self.b_size, self.b_size, 1024])
			z = relu(batch_norm(z, 'gen_norm_0'))
			z = dconv(z, 512, kernel=4, stride=2, use_bias=False, scope='dconv_1')
			z = relu(batch_norm(z, 'gen_norm1'))
			z = dconv(z, 256, kernel=4, stride=2, use_bias=False, scope='dconv_2')
			z = relu(batch_norm(z, 'gen_norm2'))
			z = dconv(z, 128, kernel=4, stride=2, use_bias=False, scope='dconv_3')
			z = relu(batch_norm(z, 'gen_norm3'))
			z = dconv(z, 64, kernel=4, stride=2, use_bias=False, scope='dconv_4')
			z = relu(batch_norm(z, 'gen_norm4'))

			z = dconv(z, self.channel, kernel=4, stride=2, use_bias=False, scope='dconv_end')
			z = tanh(z)

			return z

	def sample_z(self, b_s ,z_size):
		return	np.random.uniform(-1,1,[b_s, z_size]).astype(np.float32)

	def read_data(self, dirname, num):
		x = []
		for i in range(num):
			img = plt.imread('../'+dirname+'/'+str(i)+'.jpg')
			x.append(img[:,:,:3])
		x = np.array(x)
		x = np.reshape(x, [num, self.e_size, self.e_size, self.channel])
		x = x/127.5-1
		return x

	def next_batch(self, data, iteration):
		iteration = iteration%3
		if (iteration+1)*self.b_s > self.data_num:
			return data[iteration*self.b_s:]
		else:
			return data[iteration*self.b_s: (iteration+1)*self.b_s]

	def test(self, Pic, iteration):
		Pic = (Pic+1)/2
		for i in range(self.b_s):
			plt.subplot(4,8,i+1)
			plt.axis('off')
			plt.imshow(Pic[i])
		plt.savefig('./result_WGAN/model-'+str(iteration)+'.jpg')
		#plt.imsave('./result_0.0002/model-'+str(iteration)+'.jpg',Pic[0])
		#plt.show()


	def train(self):
		x = tf.placeholder(tf.float32, shape=[None, self.e_size, self.e_size, self.channel], name='x-input')
		z = tf.placeholder(tf.float32, shape=[None, self.z_size], name='z-input')

		G_z = self.generator(z)

		D_real, D_real_logit = self.discriminator(x, reuse=False)

		D_fake, D_fake_logit = self.discriminator(G_z, reuse=True)

		#计算GP
		GP = self.gradient_panalty(real=x, fake=G_z)

		#WGAN：D的loss增加GP项
		D_loss = discriminator_loss(D_real_logit, D_fake_logit) + GP

		G_loss = generator_loss(D_fake_logit)

		t_vars = tf.trainable_variables()
		G_vars = [var for var in t_vars if 'generator' in var.name]
		D_vars = [var for var in t_vars if 'discriminator' in var.name]

		D_Optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)

		G_Optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars)

		saver = tf.train.Saver(max_to_keep=2)

		with tf.Session() as sess:
			#init = tf.global_variables_initializer()
			#sess.run(init)
			saver.restore(sess, tf.train.latest_checkpoint('./model_WGAN/Model'))
			train_x = self.read_data(self.data_dir, self.data_num)
			for i in range(self.epoch):
				for j in range(self.iterations):
					z_sample = self.sample_z(self.b_s, self.z_size)
					x_input = self.next_batch(train_x, j)
					if j % 10 == 0:
						print("Training {} steps!".format(j+i*self.iterations+8500))
					if j % 50 == 0:
						Pic = sess.run(G_z, feed_dict={z: z_sample})
						self.test(Pic,i*self.iterations+j+8500)
					if j % 500 == 0:
						saver.save(sess, './model_WGAN/DCGAN_model_WGAN', global_step=(i*self.iterations+j+8500))
					sess.run(D_Optimizer, feed_dict={x: x_input, z: z_sample})
					sess.run(G_Optimizer, feed_dict={x: x_input, z: z_sample})
					sess.run(G_Optimizer, feed_dict={x: x_input, z: z_sample})
			#z_sample = self.sample_z(self.b_s, self.z_size)
			#Pic = sess.run(G_z, feed_dict={z: z_sample})
			#self.test(Pic)
			

def main(argv=None):
	tf.reset_default_graph()
	model = DCGAN()
	model.train()

if __name__ == '__main__':
	tf.app.run()
