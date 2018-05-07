import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
tfd = tf.contrib.distributions

mnist = input_data.read_data_sets('MNIST_data')

sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 784])

encoder_l1 = tf.layers.dense(x, 200, activation = tf.nn.relu)
encoder_l2 = tf.layers.dense(encoder_l1, 200, activation = tf.nn.relu)
loc = tf.layers.dense(encoder_l2, 2)
scale = tf.layers.dense(encoder_l2, 2, activation = tf.nn.softplus)

posterior = tfd.MultivariateNormalDiag(loc, scale)
encoded = posterior.sample()

decoder_l1 = tf.layers.dense(encoded, 200, activation = tf.nn.relu)
decoder_l2 = tf.layers.dense(decoder_l1, 200, activation = tf.nn.relu)
decoder_out = tf.layers.dense(decoder_l2, 784)
img_prob_dist = tfd.Independent(tfd.Bernoulli(decoder_out), 2)

prior = tfd.MultivariateNormalDiag(tf.zeros(2), tf.ones(2))

kl_divergence = tfd.kl_divergence(posterior, prior)
loss = -tf.reduce_mean(img_prob_dist.log_prob(x) - kl_divergence)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

sess.run(tf.global_variables_initializer())

for epoch in range(20):
    print(sess.run(loss, feed_dict = {x: mnist.test.images}))
    for _ in range(600):
      batch = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch[0]})

test = mnist.test.images.reshape(-1, 784)
x_encoded = sess.run(encoded, feed_dict={x: test})
plt.figure(figsize=(6, 6))
names = mnist.test.labels
plt.scatter(x_encoded[:, 0], x_encoded[:, 1], c = names)
plt.colorbar()
plt.show()

# fig, ax = plt.subplots(ncols = 10, nrows = 5, figsize = (10 * 1, 1))
# for row in range(5):
#     sampled_code = sess.run(prior.sample(10))
#     for sample_index in range(len(sampled_code)):
#         decoded = sess.run(img_prob_dist.mean(), feed_dict = {encoded:np.reshape(sampled_code[sample_index], [-1,2])})
#         no_ticks = dict(left='off', bottom='off', labelleft='off', labelbottom='off')
#         ax[row][sample_index].imshow(np.reshape(decoded, [28,28]), cmap='gray')
#         ax[row][sample_index].tick_params(axis='both', which='both', **no_ticks)
# plt.show()
