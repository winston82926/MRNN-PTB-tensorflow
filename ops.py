import tensorflow as tf

def _weight_variable(shape, name='weights'):
    return tf.get_variable(shape=shape, name=name)

def _bias_variable(shape, name='biases'):
    return tf.get_variable(name=name, shape=shape) 

def linear(inputs, output_dim, name=None, initializer = tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name or "linear"):
        # Get shapes
        batch_size, input_dim = inputs.get_shape().as_list()
        
        # Get variable
        W = tf.get_variable(shape=[input_dim, output_dim], name='weights', initializer = initializer)
        b = tf.get_variable(shape=[output_dim], name='biases', initializer = initializer)
        
        return tf.matmul(inputs, W) + b

def softmax_sample(logits):
    return tf.cast(tf.equal(logits,tf.reduce_max(logits,1,keep_dims=True)),logits.dtype)
    