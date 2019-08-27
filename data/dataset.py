import numpy as np
import csv
from skimage import io, transform
from copy import copy
from matplotlib import pyplot as plt
import random

class DatasetIterator:

    def __init__(self, dataset, sequence, batch_size):
        '''
            Initialize an iterator

            args:
                dataset     the dataset to traverse
                sequence    the sequence of indices that specifies in which order
                            the elements in dataset are returned
                batch_size  #samples in a batch
        '''
        self._i = 0
        self._len = len(dataset) // batch_size
        self._batch_size = batch_size
        self._dataset = dataset
        self._sequence = sequence

    def __next__(self):
        '''
            Get the next batch

            returns:
                images      the batch of images (stacked in the 0th dimension)
                labels      the batch of corresponding labels
        '''
        if self._i < self._len:
            data = [self._dataset[j] for j in self._sequence[self._i*self._batch_size:self._i*self._batch_size + self._batch_size]]
            imgs = np.expand_dims(np.array([d[0] for d in data]), -1)
            labels = np.array([d[1] for d in data])
            self._i += 1
            return imgs, labels
        raise StopIteration()

class Dataset:

    def _read_csv(self, path, labels):
        result = []
        paths = []

        with open(str(path), newline = '') as csvfile:
            reader = csv.DictReader(csvfile, delimiter = ';')

            for row in reader:
                result.append([1 if row[l] == '1' else 0 for l in labels])
                paths.append(row['filename'])

        return np.array(result, dtype = np.float32), paths

    def _load_and_normalize_images(self, base, paths):
        imgs = [transform.resize(io.imread(str(base / p), as_gray = True), (224,224)) for p in paths]
        imgs = np.array(imgs, dtype = np.float32)
        imgs -= np.mean(imgs, axis = (1,2), keepdims = True)
        imgs /= np.std(imgs, axis = (1,2), keepdims = True)
        return imgs

    def _calculate_sample_weights(self, labels):
        unique_labels, inverse_idx, unique_cnt = np.unique(labels, axis=0, return_inverse=True, return_counts=True)

        w = np.zeros(labels.shape[0], dtype=np.float64)
        for i in range(unique_labels.shape[0]):
            l = unique_labels[i].reshape(1,-1)
            idx = np.sum(np.abs(labels-l), axis = 1) == 0.0
            w[idx] = 1/unique_cnt[i]

        w /= np.sum(w)
        return w

    def _augment_image(self, img):
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)
        if random.random() < 0.5:
            img = img.T

        return img

    def __init__(self, path, label_keys, batch_size, resample, enable_augmentation):
        '''
            Initialize a dataset

            args:
                path                path to the CSV file containing labels and paths to images
                label_keys          names of labels as given in the CSV header
                batch_size          #samples that form a batch
                resample            specifies, if batches should be sampled such that labels are balanced
                                        True: return batches with balanced labels
                                        False: return samples in the order they appear in the CSV file
                enable_augmentation toggle data augmentation
                                        True: *randomly* flip (horizontally and/or vertically) or rotate (by 90 degree) images
                                        False: no augmentation
                resize_size         if not None, resize images to given size
        '''
        self._labels, image_paths = self._read_csv(path, label_keys)
        self._images = self._load_and_normalize_images(path.parent, image_paths)
        self._batch_size = batch_size
        self._len = len(self._images)
        self._resample = resample
        self._enable_augmentation = enable_augmentation
    def get_labels(self):
        return self._labels

    def __iter__(self):
        '''
            returns:
                it  a new iterator
        '''
        if self._resample:
            weights = self._calculate_sample_weights(self._labels)
            sequence = np.random.choice(np.arange(self._len), self._len, p = weights).tolist()
        else:
            sequence = np.arange(self._len).tolist()
        return DatasetIterator(self, sequence, self._batch_size)

    def __len__(self):
        '''
            returns:
                N   number of samples in the Dataset
        '''
        return self._len

    def __getitem__(self, i):
        '''
            returns:
                img     the i'th image
                label   the i'th label

            Make sure that:
                - augmentation is applied of specified
                - images are resized, if specified
                - images are normalized to have zero mean and unit variance
        '''
        img = self._augment_image(self._images[i]) if self._enable_augmentation else self._images[i]
        label = self._labels[i]
        return img, label

    def split_train_validation(self, part):
        '''
            *Randomly* split this Dataset into a train and a validation set

            args:
                part    fraction in [0,1] of images that make up the train set

            returns:
                dataset_train   a copy of this Dataset that contains only the samples used for training
                dataset_valid   a copy of this Dataset that contains only the samples used for validation
        '''
        idx = list(range(self._len))
        random.shuffle(idx)
        N_train = int(self._len*part)
        idx1 = idx[:N_train]
        idx2 = idx[N_train:]
        idx1 = [i for i in range(self._len) if i not in idx2]
        d1 = copy(self)
        d2 = copy(self)
        d1._images = self._images[idx1]
        d1._labels = self._labels[idx1]
        d1._len = len(idx1)
        d2._images = self._images[idx2]
        d2._labels = self._labels[idx2]
        d2._len = len(idx2)
        d2._enable_augmentation = False
        d2._resample = False

        return d1, d2

    @property
    def image_shape(self):
        '''
            The shape (w, h) of samples
        '''
        return self._images[0].shape

    @property
    def label_shape(self):
        '''
            The length of each label vector
        '''
        return self._labels[0].shape

