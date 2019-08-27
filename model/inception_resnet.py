import tensorflow as tf
from tensorflow import keras

SQUEEZE_RATIO   = 0.25
DROPOUT         = 0.2
#regularizer = None
#fc_regularizer = None
initializer = tf.contrib.layers.variance_scaling_initializer()
bias_initializer = tf.constant_initializer(0.1)
LOOPS = 5
regularizer = keras.regularizers.l2(2e-1)
fc_regularizer = keras.regularizers.l2(3e-4)
# initializer = tf.contrib.layers.variance_scaling_initializer()
def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''

    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)

    # TODO
    with tf.variable_scope('inception_resnet', reuse=tf.AUTO_REUSE):
        output = stem(x)
        for i in range(LOOPS):
            output = inception_res_A(output,filters=256,name='I_R_A_'+str(i))

        
        output1 = tf.reduce_mean(output,axis=[1,2],keep_dims=False,name='Global_Average_Pooling')
        output1 = tf.layers.dropout(output1,DROPOUT,training=is_training,name='Dropout')

        output1 = tf.layers.dense(output1,num_outputs,kernel_initializer=initializer,bias_initializer=bias_initializer,kernel_regularizer=regularizer,name='FC_3')

        #output = seNet(output,256,is_training=is_training,name='SeNet_1')

        output = reduction(output,filters=256,name='Reduce_1')
        for i in range(2*LOOPS):
            output = inception_res_B(output,filters=512,name='I_R_B_'+str(i))

        #output = seNet(output,512,is_training=is_training,name='SeNet_2')

        output = reduction(output,filters=512,name='Reduce_2')

        for i in range(LOOPS):
            output = inception_res_C(output,filters=1024,name='I_R_C_'+str(i))

        output = tf.reduce_mean(output,axis=[1,2],keep_dims=False,name='Global_Average_Pooling')
        output = tf.layers.dropout(output,DROPOUT,training=is_training,name='Dropout')

        output = tf.layers.dense(output,num_outputs,kernel_initializer=initializer,bias_initializer=bias_initializer,kernel_regularizer=regularizer,name='FC_3')

    return output,output1



def stem(input):
    with tf.variable_scope('Stem', reuse=tf.AUTO_REUSE):
        output = conv2d(input=input,filters=32,kernel_size=3,strides=2,padding="valid",name='Conv_1')
        output = conv2d(output,filters=32,kernel_size=3,padding="valid",name='Conv_2')
        output = conv2d(output,filters=64,kernel_size=3,padding="same",name='Conv_3')

        output = tf.layers.max_pooling2d(output,pool_size=3,strides=2,padding='valid',name='Pool_1')

        output = conv2d(output,filters=80,kernel_size=1,padding="same",name='Conv4')
        output = conv2d(output,filters=192,kernel_size=3,padding="valid",name='Conv5')
        output = conv2d(output,filters=256,kernel_size=3,strides=2,padding="valid",name='Conv6')
        # output = tf.layers.batch_normalization(output,name='Stem_BN2')
        # output = tf.nn.relu(output,name='Stem_ReLU')

        return output


def reduction(input,filters,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filter = int(filters/2)
        output1 = conv2d(input,filters=filter,kernel_size=1,padding="same",name='Conv_1')
        output1 = conv2d(output1,filters=filter,kernel_size=3,padding="same",name='Conv_2')
        output1 = conv2d(output1,filters=filter,kernel_size=3,strides=2,padding="same",name='Conv_3')

        output2 = conv2d(input,filters=filter,kernel_size=1,padding="same",name='onv4')
        output2 = conv2d(output2,filters=filter,kernel_size=3,strides=2,padding="same",name='Conv5')

        output3 = tf.layers.max_pooling2d(input,pool_size=3,strides=2,padding='same',name='Pool_1')

        output = [output1,output2,output3]
        output = tf.concat(output, axis=3)

        output = tf.layers.batch_normalization(output,axis=3,name='BN')
        output = tf.nn.relu(output,name='ReLU')

        return output


def inception_res_A(input,filters,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output1 = conv2d(input,filters=32,kernel_size=1,padding="same",name='Conv_1')

        output2 = conv2d(input,filters=32,kernel_size=1,padding="same",name='Conv_21')
        output2 = conv2d(output2,filters=32,kernel_size=3,padding="same",name='Conv_22')

        output3 = conv2d(input,filters=32,kernel_size=1,padding="same",name='Conv_31')
        output3 = conv2d(output3,filters=32,kernel_size=3,padding="same",name='Conv_32')
        output3 = conv2d(output3,filters=32,kernel_size=3,padding="same",name='Conv_33')

        output = conv2d(tf.concat([output1,output2,output3],axis=3),filters=filters,kernel_size=1,padding="same",name='Conv_4')
        output = output + input

        output = tf.layers.batch_normalization(output,axis=3,name='BN')
        output = tf.nn.relu(output,name='ReLU')

        return output


def inception_res_B(input,filters,name=''):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        output1 = conv2d(input,filters=192,kernel_size=1,padding="same",name='Conv_1')

        output2 = conv2d(input,filters=128,kernel_size=1,padding="same",name='Conv_21')
        output2 = conv2d(output2,filters=160,kernel_size=(1,7),padding="same",name='Conv_22')
        output2 = conv2d(output2,filters=192,kernel_size=(7,1),padding="same",name='Conv_23')

        output = conv2d(tf.concat([output1,output2],axis=3),filters=filters,kernel_size=1,padding="same",name='Conv_3')

        output = output + input

        output = tf.layers.batch_normalization(output,axis=3,name='BN')
        output = tf.nn.relu(output,name='ReLU')

        return output


def inception_res_C(input,filters,name=''):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        output1 = conv2d(input,filters=192,kernel_size=1,padding="same",name='Conv_1')

        output2 = conv2d(input,filters=192,kernel_size=1,padding="same",name='Conv_21')
        output2 = conv2d(output2,filters=224,kernel_size=(1,3),padding="same",name='Conv_22')
        output2 = conv2d(output2,filters=256,kernel_size=(3,1),padding="same",name='Conv_23')

        output = conv2d(tf.concat([output1,output2],axis=3),filters=filters,kernel_size=1,padding="same",name='Conv_3')

        output = output + input

        output = tf.layers.batch_normalization(output,axis=3,name='BN')
        output = tf.nn.relu(output,name='ReLU')

        return output

def seNet(input,filters,ratio=SQUEEZE_RATIO,initializer=tf.contrib.layers.variance_scaling_initializer(),regularizer=fc_regularizer,is_training=False,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        squeeze = tf.reduce_mean(input,axis=[1,2],keep_dims=False,name='Global_Average_Pooling')
        excitation = tf.layers.dense(squeeze,units=filters*ratio,kernel_initializer=initializer,kernel_regularizer=regularizer,name='SE_FC1')
        excitation = tf.layers.dropout(excitation,0.5,training=is_training)
        excitation = tf.nn.relu(excitation,name='SE_ReLU')
        excitation = tf.layers.dense(excitation,units=filters,kernel_initializer=initializer,kernel_regularizer=regularizer,name='SE_FC2')
        excitation = tf.nn.sigmoid(excitation,name='SE_Sigmoid')
        excitation = tf.reshape(excitation,[-1,1,1,filters])
        output = excitation * input
    return output

def conv2d(input,
           filters,
           kernel_size,
           padding="same",
           strides=1,
           kernel_initializer=initializer,
           kernel_regularizer=regularizer,
           name=''):
    output = tf.layers.conv2d(input,filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,name=name)
    output = tf.layers.batch_normalization(output,axis=3,name=name+'_BN')
    output = tf.nn.relu(output,name=name+'_ReLU')

    return output
