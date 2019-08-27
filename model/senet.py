import tensorflow as tf
from tensorflow import keras
SQUEEZE_RATIO   = 0.25
regularizer = keras.regularizers.l2(2e-2)
initializer = tf.contrib.layers.variance_scaling_initializer()
fc_regularizer = keras.regularizers.l2(1e-4)
def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''

    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)

    # TODO
    with tf.variable_scope('ResNet-Modified', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(x,filters=32,kernel_size=(7,1),kernel_initializer=initializer,strides=(2,1),kernel_regularizer=regularizer,name='Conv_11')
        conv1 = tf.layers.conv2d(conv1,filters=32,kernel_size=(1,7),kernel_initializer=initializer,strides=(1,2),kernel_regularizer=regularizer,name='Conv_12')
        #conv1 = tf.layers.conv2d(conv1,filters=32,kernel_size=3,kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=keras.regularizers.l2(LAMBDA),name='Conv13')
        maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=3,strides=2,padding='same',name='Max_Pool_1')
        bn1 = tf.layers.batch_normalization(maxpool1,name='BN_1')
        relu1 = tf.nn.relu(bn1,name='ReLU_1')
        conv2 = tf.layers.conv2d(relu1,filters=64,kernel_size=3,strides=1,kernel_initializer=initializer,kernel_regularizer=regularizer,padding='same',name='Conv_21')
        maxpool2 = tf.layers.max_pooling2d(conv2,pool_size=3,strides=2,padding='same',name='Max_Pool_2')
        bn2 = tf.layers.batch_normalization(maxpool2,name="BN_2")
        relu2 = tf.nn.relu(bn2,name='ReLU_2')
        resBlock = ResBlock(input=relu2,channels=64,stride=1,name='ResBlock_1',is_training=is_training)
        resBlock = seNet(resBlock,filters=4*64,is_training=is_training,name='SENET_1')
        resBlock = ResBlock(resBlock,128,2,name='ResBlock_2',is_training=is_training)
        #resBlock = tf.layers.max_pooling2d(resBlock,pool_size=3,strides=2,padding='same',name="Max_Pool_2")
        #resBlock = seNet(resBlock,filters=4*128,is_training=is_training,name='SENET_2')
        resBlock = ResBlock(resBlock,256,2,name='ResBlock_3',is_training=is_training)
        #resBlock = tf.layers.average_pooling2d(resBlock,pool_size=3,strides=2,padding='same',name='Ave_Pool')
        resBlock = seNet(resBlock,filters=4*256,is_training=is_training,name='SENET_3')
        resBlock = ResBlock(resBlock,512,1,name='ResBlock_4',is_training=is_training)
        output = tf.reduce_mean(resBlock,axis=[1,2],keep_dims=False,name='Global_Average_Pooling')
        #output = tf.layers.dropout(output,0.2,training=is_training)
        #output = tf.layers.dense(output,units=4096,kernel_initializer=initializer,name='FC0')
        
        result = tf.layers.dense(output,units=num_outputs,kernel_initializer=initializer,kernel_regularizer=fc_regularizer,name='FC')

        return result

    pass

def ResBlock(input,channels,stride,name='',initializer=tf.contrib.layers.variance_scaling_initializer(),regularizer=regularizer,is_training=False):
    blocklist = []
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('5by5_Filter', reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(input,channels,kernel_size=3,padding='same',kernel_initializer=initializer,strides=stride,kernel_regularizer=regularizer,name='Conv_11')
            output1 = tf.layers.conv2d(conv,channels,kernel_size=3,padding='same',kernel_initializer=initializer,kernel_regularizer=regularizer,name='Conv_12')

            output1 = tf.layers.batch_normalization(inputs=output1,name='BN')
            output1 = tf.nn.relu(output1,name='ReLU')

            blocklist.append(output1)
        with tf.variable_scope('3by3_Filter', reuse=tf.AUTO_REUSE):
            output2 = tf.layers.conv2d(input,channels,kernel_size=3,padding='same',kernel_initializer=initializer,strides=stride,kernel_regularizer=regularizer,name='Conv_11')

            output2 = tf.layers.batch_normalization(inputs=output2,name='BN')
            output2 = tf.nn.relu(output2,name='ReLU')
            
            blocklist.append(output2)
        with tf.variable_scope('maxpool', reuse=tf.AUTO_REUSE):
            maxpool = tf.layers.max_pooling2d(input,pool_size=3,padding='same',strides=stride,name=name+'_MaxPool')
            output3 = tf.layers.conv2d(maxpool,channels,kernel_size=1,padding='same',kernel_initializer=initializer,kernel_regularizer=regularizer,name='Conv')
            output3 = tf.layers.batch_normalization(output3,name='BN')
            output3 = tf.nn.relu(output3,name='ReLU')

            blocklist.append(output3)
        with tf.variable_scope('1by1_Filter', reuse=tf.AUTO_REUSE):
            output4 = tf.layers.conv2d(input,channels,kernel_size=1,padding='same',kernel_initializer=initializer,strides=stride,kernel_regularizer=regularizer,name='Conv')

            output4 = tf.layers.batch_normalization(inputs=output4,name='BN')
            output4 = tf.nn.relu(output4,name='ReLU')

            blocklist.append(output4)
        #with tf.variable_scope('1by1_Filter_2', reuse=tf.AUTO_REUSE):
        #    conv = tf.layers.conv2d(input,len(blocklist)*channels,kernel_size=1,padding='same',kernel_initializer=initializer,strides=stride,kernel_regularizer=regularizer,name='Conv')

        #    output5 = tf.layers.batch_normalization(inputs=conv,name='BN')

        output = tf.concat(blocklist, axis=3)
        # tf.summary.image("image"+name, tensor=, max_outputs=1)
        #bn = tf.layers.batch_normalization(output,name='BN')
        #output = tf.nn.relu(bn,name='ReLU')
        #output = tf.add(output,output5)
    return output

def seNet(input,filters,ratio=SQUEEZE_RATIO,initializer=tf.contrib.layers.variance_scaling_initializer(),regularizer=regularizer,is_training=False,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        squeeze = tf.reduce_mean(input,axis=[1,2],keep_dims=False,name='Global_Average_Pooling')
        excitation = tf.layers.dense(squeeze,units=filters*ratio,kernel_initializer=initializer,kernel_regularizer=regularizer,name='SE_FC1')
        excitation = tf.layers.dropout(excitation,0.5,training=is_training)
        excitation = tf.nn.relu(excitation,name='SE_ReLU')
        excitation = tf.layers.dense(excitation,units=filters,kernel_initializer=initializer,kernel_regularizer=regularizer,name='SE_FC2')
        #excitation = tf.layers.dropout(excitation,0.2,training=is_training)
        excitation = tf.nn.sigmoid(excitation,name='SE_Sigmoid')
        excitation = tf.reshape(excitation,[-1,1,1,filters])
        output = excitation * input
    return output
