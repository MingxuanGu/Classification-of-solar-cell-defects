import numpy as np
from . import measures

class Evaluation:

    def __init__(self, measures):
        '''
            Init the evaluation using the given measures

            args:
                measures    list of ClasswiseMeasure instances
        '''
        self._measures = measures

    def add_batch(self, predictions, labels):
        '''
            Add a batch to the evaluation by passing it to all measures

            args:
                predictions     np.ndarray of predictions made by the network
                labels          np.ndarray of corresponding ground truth labels
        '''
        for m in self._measures:
            m.add_batch(predictions, labels)

    def flush(self):
        '''
            Prints all measures to the console and clears them
        '''
        for m in self._measures:
            print('{}:'.format(m.name))
            for l, v in zip(m.labels(), m.values()):
                print('\t{}: {:.2f}'.format(l, v))
            m.clear()


def create_evaluation(measure_names, class_names):
    '''
        Factory method that should be used to create and evaluation

        args:
            measure_names   list of strings of measure names, e.g. ['ClasswiseAccuracy', 'ClasswiseF1']
            class_names     list of class names in the order in which they appear in the label vectors
    '''
    m = []
    for mn in measure_names:
        class_ = getattr(measures, mn)
        m.append(class_(class_names))
    return Evaluation(m)


