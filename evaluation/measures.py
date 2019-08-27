from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class ClasswiseMeasure(ABC):

    @abstractmethod
    def _new_batch(labels, predictions):
        '''
            Called by new_batch after type casting is applied
        '''
        pass

    def __init__(self, class_names):
        '''
            Initialize this measure

            args:
                class_names     list of class names as they appear in the labels
        '''
        self._class_names = class_names

    def add_batch(self, predictions, labels):
        '''
            Add a new batch

            args:
                predictions     np.ndarray with predictions by the network
                labels          np.ndarray with corresponding ground truth labels
        '''
        predictions = predictions.astype(np.int32)
        labels = labels.astype(np.int32)
        self._new_batch(labels, predictions)

        if self._class_names is None:
            self._class_names = [str(x) for x in range(predictions.shape[-1])]

    def labels(self):
        '''
            returns:
                class_names
        '''
        return self._class_names

    @abstractmethod
    def values(self):
        '''
            Calculates the measure for each class given all previsous batches

            returns:
                values  a list of results
        '''
        pass

    def clear(self):
        '''
            Reset this measure to the initial state
        '''
        self._labels = None
        self._predictions = None

    @abstractproperty
    def name(self):
        '''
            An identifying name for this measure
        '''
        pass

    @staticmethod
    def safe_divide(x, y):
        '''
            Divides x by y avoiding division by zero erros
        '''
        if isinstance(y, np.ndarray):
            zeros = y == 0.0
            y[zeros] = 1
            res = x/y
            res[zeros] = 0.0
            return res
        else:
            if y == 0:
                if isinstance(x, np.ndarray):
                    return np.zeros_like(x).astype(np.float64)
                else:
                    return 0.0
            else:
                return x/y

class ClasswiseAccuracy(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count = 0
        self._count_correct = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count = 0
        self._count_correct = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count += labels.shape[0]
        self._count_correct += np.sum(predictions == labels, axis = 0)

    def values(self):
        return (self.safe_divide(self._count_correct, self._count)).tolist()

    @property
    def name(self):
        return 'classwise_accuracy'

class ClasswiseRecall(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_correct_pos = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_pos = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_correct_pos = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_pos = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_correct_pos += np.sum(np.logical_and(predictions == labels, labels), axis = 0)
        self._count_pos += np.sum(labels, axis = 0)

    def values(self):
        return (self.safe_divide(self._count_correct_pos, self._count_pos)).tolist()

    @property
    def name(self):
        return 'classwise_recall'

class ClasswisePrecision(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_correct_pos = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_labels = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_correct_pos = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_labels = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_correct_pos += np.sum(np.logical_and(predictions == labels, labels), axis = 0)
        self._count_labels += np.sum(np.logical_or(labels, predictions), axis = 0)

    def values(self):
        return (self.safe_divide(self._count_correct_pos, self._count_labels)).tolist()

    @property
    def name(self):
        return 'classwise_precision'

class ClasswiseSpecificity(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_correct_neg = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_nolabels = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_correct_neg = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_nolabels = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_correct_neg += np.sum(np.logical_and(predictions == labels, np.logical_not(labels)), axis = 0)
        self._count_nolabels += np.sum(np.logical_not(labels), axis = 0)

    def values(self):
        return (self.safe_divide(self._count_correct_neg, self._count_nolabels)).tolist()

    @property
    def name(self):
        return 'classwise_specificity'

class ClasswiseF1(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_correct_pos = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_labels = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_real = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_correct_pos = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_labels = np.zeros(len(self._class_names), dtype = np.int32)
        self._count_real = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_correct_pos += np.sum(np.logical_and(predictions == labels, labels), axis = 0)
        self._count_labels += np.sum(np.logical_or(labels, predictions), axis = 0)
        self._count_real += np.sum(labels, axis = 0)

    def values(self):
        precision = self.safe_divide(self._count_correct_pos, self._count_labels)
        recall = self.safe_divide(self._count_correct_pos, self._count_real)

        return (self.safe_divide(2*precision*recall, (precision+recall))).tolist()

    @property
    def name(self):
        return 'classwise_f1'


class ClasswiseConfusionMatrixTP(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_tp = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_tp = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_tp += np.sum(np.logical_and(predictions == labels, labels), axis = 0)

    def values(self):
        return self._count_tp.tolist()

    @property
    def name(self):
        return 'classwise_confusion_tp'

class ClasswiseConfusionMatrixTN(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_tn = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_tn = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_tn += np.sum(np.logical_and(predictions == labels, np.logical_not(labels)), axis = 0)

    def values(self):
        return self._count_tn.tolist()

    @property
    def name(self):
        return 'classwise_confusion_tn'

class ClasswiseConfusionMatrixFP(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_fp = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_fp = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_fp += np.sum(np.logical_and(predictions != labels, np.logical_not(labels)), axis = 0)

    def values(self):
        return self._count_fp.tolist()

    @property
    def name(self):
        return 'classwise_confusion_fp'

class ClasswiseConfusionMatrixFN(ClasswiseMeasure):

    def __init__(self, class_names):
        super().__init__(class_names)

        self._count_fn = np.zeros(len(self._class_names), dtype = np.int32)

    def clear(self):
        super().clear()

        self._count_fn = np.zeros(len(self._class_names), dtype = np.int32)

    def _new_batch(self, labels, predictions):
        self._count_fn += np.sum(np.logical_and(predictions != labels, labels), axis = 0)

    def values(self):
        return self._count_fn.tolist()

    @property
    def name(self):
        return 'classwise_confusion_fn'
