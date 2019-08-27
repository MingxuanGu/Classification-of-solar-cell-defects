import tensorflow as tf
from tensorflow import keras

LAMBDA = 0.005
def create(x, num_outputs, dropout_rate = 0.5):
    '''
        args:
            x               network input
            num_outputs     number of logits
            dropout         dropout rate during training
    '''

    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)

    # TODO
    initializer = tf.contrib.layers.xavier_initializer()
    x = tf.layers.max_pooling2d(x,pool_size=10,strides=2,name='max_pooling')
    input = tf.layers.flatten(x)
    fc1 = tf.layers.dense(input,4096,activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='FC_1')
    dropout2 = tf.layers.dropout(fc1,dropout_rate,training=is_training,name='Dropout_2')
    fc2 = tf.layers.dense(dropout2,4096,activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='FC_2')
    output = tf.layers.dense(fc2,num_outputs,kernel_initializer=initializer,name='FC_3')

    return output

    # with tf.variable_scope('testnet',reuse=tf.AUTO_REUSE):
    #     # block1 = CBRblock(input=x,filter_num=96,filter_size=11,conv_stride=4,pooling=True,pool_size=3,pool_stride=2,name='Block1')
    #     # block2 = CBRblock(block1,192,5,1,True,3,2,name='Block2')
    #     # block3 = CBRblock(block2,384,3,1,name='Block3')
    #     # block4 = CBRblock(block3,256,3,1,name='Block4')
    #     # block5 = CBRblock(block4,256,3,1,True,3,2,name='Block5')
    #     # dropout1 = tf.layers.dropout(block5,dropout_rate,training=is_training,name='Dropout_1')
    #     # input = tf.layers.flatten(dropout1)
    #     x = tf.layers.max_pooling2d(x,pool_size=10,strides=2,name='max_pooling')
    #     input = tf.layers.flatten(x)
    #     fc1 = tf.layers.dense(input,4096,activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='FC_1')
    #     dropout2 = tf.layers.dropout(fc1,dropout_rate,training=is_training,name='Dropout_2')
    #     fc2 = tf.layers.dense(dropout2,4096,activation=tf.nn.relu,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='FC_2')
    #     output = tf.layers.dense(fc2,num_outputs,kernel_initializer=initializer,name='FC_3')

        # return output


    pass

def CBRblock(input,filter_num,filter_size,conv_stride,pooling=False,pool_size=None,pool_stride=None,padding='valid',
             initializer = tf.contrib.layers.xavier_initializer(),name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(inputs=input,filters=filter_num,kernel_size=filter_size,strides=conv_stride,padding=padding,
                                kernel_regularizer=keras.regularizers.l2(LAMBDA), kernel_initializer=initializer,name=name+'_Conv')
        bn = tf.layers.batch_normalization(conv,beta_regularizer=keras.regularizers.l2(LAMBDA),name=name+'_BN')
        relu = tf.nn.relu(bn,name=name+'_ReLU')
        if pooling:
            output = tf.layers.max_pooling2d(relu,pool_size=pool_size,strides=pool_stride,name=name+'_MaxPool')
            return output
        else:
            return relu
