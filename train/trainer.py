import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
FALSE = tf.Variable(initial_value=False, trainable=False)
TRUE = tf.Variable(initial_value=True, trainable=False)
class Trainer:

    def __init__(self, loss, predictions, optimizer, ds_train, ds_validation, stop_patience, evaluation, inputs, labels):
        '''
            Initialize the trainer

            Args:
                loss        	an operation that computes the loss
                predictions     an operation that computes the predictions for the current
                optimizer       optimizer to use
                ds_train        instance of Dataset that holds the training data
                ds_validation   instance of Dataset that holds the validation data
                stop_patience   the training stops if the validation loss does not decrease for this number of epochs
                evaluation      instance of Evaluation
                inputs          placeholder for model inputs
                labels          placeholder for model labels
        '''
        self._train_op = optimizer.minimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._ds_train = ds_train
        self._ds_validation = ds_validation
        self._stop_patience = stop_patience
        self._evaluation = evaluation
        self._validation_losses = []
        self._model_inputs = inputs
        self._model_labels = labels
        self._train_loss = []
        self._validation_f1 = []

        with tf.variable_scope('model', reuse = True):
            self._model_is_training = tf.get_variable('is_training', dtype = tf.bool, trainable=False)


    def _train_epoch(self, sess):
        '''
            trains for one epoch and prints the mean training loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''


        # TODO
        self._model_is_training = tf.identity(TRUE, name="is_training")
        dsIter = self._ds_train.__iter__()
        mean_loss = 0
        for i in range(dsIter._len):
            images,labels = next(dsIter)
            _,loss_value,prediction = sess.run([self._train_op,self._loss,self._predictions],feed_dict={self._model_inputs:images,self._model_labels:labels})
            mean_loss = mean_loss + loss_value
            self._evaluation.add_batch(prediction,labels)

        mean_loss = mean_loss/dsIter._len
        self._train_loss.append(mean_loss)
        print("training epoch:")
        self._evaluation.flush()
        print("mean loss: ",mean_loss)

        pass

    def _valid_step(self, sess):
        '''
            run the validation and print evalution + mean validation loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''

        # TODO
        self._model_is_training = tf.identity(FALSE, name="is_validating")
        dsIter = self._ds_validation.__iter__()
        mean_loss = 0

        for i in range(dsIter._len):
            imgs,labels = next(dsIter)
            prediction,loss_value = sess.run([self._predictions,self._loss],feed_dict={self._model_inputs:imgs,self._model_labels:labels})

            mean_loss = mean_loss + loss_value
            self._evaluation.add_batch(prediction,labels)

        mean_loss = mean_loss / dsIter._len
        self._validation_losses.append(mean_loss)
        f1values = self._evaluation._measures[-1].values()
        f1mean = (f1values[0]+f1values[1]) / 2
        self._validation_f1.append(f1mean)
        print("validation step:")
        self._evaluation.flush()
        print("validation_loss: ",mean_loss)
        print("F1mean: ",f1mean)

        pass

    def _should_stop(self):
        '''
            determine if training should stop according to stop_patience
        '''

        # TODO
        # if len(self._validation_losses)<self._stop_patience:
        #     return False
        # else:
        #     query = self._validation_losses[-self._stop_patience:]
        #     if any(query[i+1] < query[i] for i in range(0,len(query)-1)):
        #         return False
        #     else:
        #         return True
        if len(self._validation_f1)<self._stop_patience:
            return False
        else:
            # if self._validation_f1[-1] > 0.8:
            #     return True
            query = self._validation_f1[-self._stop_patience:]
            if any(query[i+1] > query[i] for i in range(0,len(query)-1)):
                return False
            else:
                return True



        pass

    def run(self, sess, num_epochs=-1):
        '''
            run the training until num_epochs exceeds or the validation loss did not decrease
            for stop_patience epochs

            args:
                sess        the tensorflow session that should be used
                num_epochs  limit to the number of epochs, -1 means not limit
        '''

        # initial validation step
        self._valid_step(sess)

        i = 0

        # training loop
        while i < num_epochs or num_epochs == -1:
            print("epochs:{:d}".format(i))
            self._train_epoch(sess)
            self._valid_step(sess)
            i += 1
            # if i % 30 == 0:
            #     plot = np.round(self._train_loss.copy(), decimals=5)
            #     plt.plot(plot[1:], color="blue")
            #     plot = np.round(self._validation_losses.copy(), decimals=5)
            #     plt.plot(plot[1:], color="red")
            #     plot = np.round(self._validation_f1.copy(), decimals=5)
            #     plt.plot(plot[1:], color="green")
            #     plt.legend(['training', 'validation','validation_f1'], loc='upper left')
            #     plt.show()

            if self._should_stop():
                break

        print("end of run")
        print("start to plot")
        # plot = np.round(self._train_loss.copy(), decimals=5)
        # plt.plot(plot[1:])
        # plot = np.round(self._validation_losses.copy(), decimals=5)
        # plt.plot(plot[1:])
        # plot = np.round(self._validation_f1.copy(), decimals=5)
        # plt.plot(plot[1:], color="green")
        # plt.legend(['training', 'validation','validation_f1'], loc='upper left')
        # plt.show()










