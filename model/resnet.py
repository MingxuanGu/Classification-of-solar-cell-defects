import tensorflow as tf
from tensorflow import keras
LAMBDA = 0.005
def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''
    
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    
    # TODO
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('ResNet-18', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(x,filters=64,kernel_size=7,kernel_initializer=initializer,strides=2,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='Conv_1')
        bn1 = tf.layers.batch_normalization(conv1,name='BN_1')
        relu1 = tf.nn.relu(bn1,name='ReLU_1')
        maxpool1 = tf.layers.max_pooling2d(relu1,pool_size=3,strides=2,name='Max_Pool_1')
        resBlock1 = ResBlock(maxpool1,64,1,name='ResBlock_1')
        resBlock2 = ResBlock(resBlock1,128,2,name='ResBlock_2')
        resBlock3 = ResBlock(resBlock2,256,2,name='ResBlock_3')
        resBlock4 = ResBlock(resBlock3,512,2,name='ResBlock_4')
        avgpool = tf.reduce_mean(resBlock4,axis=[1,2],keepdims=False,name='Global_Average_Pooling')
        fc = tf.layers.dense(avgpool,units=num_outputs,kernel_initializer=initializer,name='FC')

        return fc

    pass

def ResBlock(input,channels,stride,name='',intializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # print(input)
        conv1 = tf.layers.conv2d(input,channels,kernel_size=3,padding='same',kernel_initializer=intializer,strides=stride,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='Conv_1')
        # print(conv1)
        bn1 = tf.layers.batch_normalization(conv1,name='BN_1')
        relu1 = tf.nn.relu(bn1,name='ReLU_1')
        conv2 = tf.layers.conv2d(relu1,channels,kernel_size=3,padding='same',kernel_initializer=intializer,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='Conv_2')
        # print(conv2)
        bn2 = tf.layers.batch_normalization(conv2,name='BN_2')
        relu2 = tf.nn.relu(bn2,name='ReLU_2')

        conv = tf.layers.conv2d(input,channels,kernel_size=1,padding='same',kernel_initializer=intializer,strides=stride,kernel_regularizer=keras.regularizers.l2(LAMBDA),name='Conv')
        bn = tf.layers.batch_normalization(conv,name='BN')
        output = tf.add(relu2,bn)

    return output

