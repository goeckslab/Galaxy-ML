from keras.preprocessing.image import ImageDataGenerator
from sklearn.base import BaseEstimator


class ImageBatchGenerator(ImageDataGenerator, BaseEstimator):
    pass
