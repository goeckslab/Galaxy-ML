import numpy as np
import os
import warnings

from keras_preprocessing.image import (DataFrameIterator
                                       as KerasDataFrameIterator)
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.utils import validate_filename
from keras.utils.data_utils import Sequence
from sklearn.base import BaseEstimator


allowed_class_modes = {
    'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse', None}

white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')


class DataFrameIterator(KerasDataFrameIterator, BaseEstimator, Sequence):
    """ Override `keras_preprocessing.image.DataFrameIterator`

    Parameters
    -----------

    """
    def __init__(self, dataframe,
                 directory=None,
                 image_data_generator=None,
                 x_col="filename",
                 y_col="class",
                 weight_col=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 interpolation='nearest',
                 dtype='float32'):

        super(DataFrameIterator, self).set_processing_attrs(
            image_data_generator, target_size, color_mode,
            data_format, save_to_dir, save_prefix,
            save_format, interpolation)

        df = dataframe.copy()
        self.directory = directory or ''
        self.class_mode = class_mode
        self.dtype = dtype
        if class_mode not in ["input", "multi_output", "raw", None]:
            num_classes = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))
        if class_mode not in ["input", "multi_output", "raw", None]:
            self.classes = self.get_classes(df, y_col)
        self.filenames = df[x_col].tolist()
        self._sample_weight = df[weight_col].values if weight_col else None
        if class_mode == "multi_output":
            self._targets = [np.array(df[col].tolist()) for col in y_col]
        if class_mode == "raw":
            self._targets = df[y_col].values
        self.samples = len(self.filenames)
        if class_mode in ["input", "multi_output", "raw", None]:
            print('Found {} image filenames.'
                  .format(self.samples))
        else:
            print('Found {} image filenames belonging to {} classes.'
                  .format(self.samples, num_classes))
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super(DataFrameIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)


class ImageDataFrameBatchGenerator(ImageDataGenerator, BaseEstimator):
    """ Extend `keras_preprocessing.image.ImageDataGenerator` to work with
    DataFrameIterator exclusively, generating batches of tensor data from
    images with online augumentation.

    Parameters
    ----------
    From `keras_preprocessing.image.ImageDataGenerator`.
    featurewise_center : Boolean.
        Set input mean to 0 over the dataset, feature-wise.
    samplewise_center : Boolean. Set each sample mean to 0.
    featurewise_std_normalization : Boolean.
        Divide inputs by std of the dataset, feature-wise.
    samplewise_std_normalization : Boolean. Divide each input by its std.
    zca_whitening : Boolean. Apply ZCA whitening.
    zca_epsilon : epsilon for ZCA whitening. Default is 1e-6.
    rotation_range : Int. Degree range for random rotations.
    width_shift_range : Float, 1-D array-like or int.
    height_shift_range : Float, 1-D array-like or int.
    brightness_range : Tuple or list of two floats.
    shear_range : Float. Shear Intensity.
    zoom_range : Float or [lower, upper].
    channel_shift_range : Float. Range for random channel shifts.
    fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
        Default is 'nearest'.
        Points outside the boundaries of the input are filled
        according to the given mode:
        - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
        - 'nearest':  aaaaaaaa|abcd|dddddddd
        - 'reflect':  abcddcba|abcd|dcbaabcd
        - 'wrap':  abcdabcd|abcd|abcdabcd
    cval : Float or Int.
    horizontal_flip : Boolean.
        Randomly flip inputs horizontally.
    vertical_flip : Boolean.
    rescale : rescaling factor. Defaults to None.
    preprocessing_function : function that will be applied on each input.
        The function will run after the image is resized and augmented.
        The function should take one argument:
        one image (Numpy tensor with rank 3),
        and should output a Numpy tensor with the same shape.
    data_format : Image data format,
        either "channels_first" or "channels_last".
        "channels_last" mode means that the images should have shape
        `(samples, height, width, channels)`,
        "channels_first" mode means that the images should have shape
        `(samples, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    interpolation_order : Int.
    dtype : Dtype to use for the generated arrays. Default is 'float32'.

    From `keras_preprocessing.image.ImageDataGenerator.flow_from_dataframe`.
    dataframe : Pandas dataframe containing the filepaths relative to
        `directory`.
    directory: string, path to the directory to read images from. If `None`,
        data in `x_col` column should be absolute paths.
    x_col: string, column in `dataframe` that contains the filenames (or
                absolute paths if `directory` is `None`).
    y_col: string or list, column/s in `dataframe` that has the target data.
    weight_col: string, column in `dataframe` that contains the sample
        weights. Default: `None`.
    target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
        The dimensions to which all images found will be resized.
    color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to have 1 or 3 color channels.
    classes: optional list of classes (e.g. `['dogs', 'cats']`).
        Default: None. If not provided, the list of classes will be
        automatically inferred from the `y_col`,
        which will map to the label indices, will be alphanumeric).
        The dictionary containing the mapping from class names to class
        indices can be obtained via the attribute `class_indices`.
    class_mode: one of "binary", "categorical", "input", "multi_output",
        "raw", sparse" or None. Default: "categorical".
        Mode for yielding the targets:
        - `"binary"`: 1D numpy array of binary labels,
        - `"categorical"`: 2D numpy array of one-hot encoded labels.
            Supports multi-label output.
        - `"input"`: images identical to input images (mainly used to
            work with autoencoders),
        - `"multi_output"`: list with the values of the different columns,
        - `"raw"`: numpy array of values in `y_col` column(s),
        - `"sparse"`: 1D numpy array of integer labels,
        - `None`, no targets are returned (the generator will only yield
            batches of image data, which is useful to use in
            `model.predict_generator()`).
    batch_size: size of the batches of data (default: 32).
    shuffle: whether to shuffle the data (default: True)
    seed: optional random seed for shuffling and transformations.
    interpolation: Interpolation method used to resample the image if the
        target size is different from that of the loaded image.
        Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
        If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
        supported. If PIL version 3.4.0 or newer is installed, `"box"` and
        `"hamming"` are also supported. By default, `"nearest"` is used.
    validate_filenames: Boolean, whether to validate image filenames in
        `x_col`. If `True`, invalid images will be ignored. Disabling this
        option can lead to speed-up in the execution of this function.
        Default: `True`.
    fit_sample_size : Int. Default is None / 1000.
        Number of training images used in `datagen.fit`.
        Relevant only when `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening are set` are set
        to True.
    """
    def __init__(self, dataframe,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-06,
                 rotation_range=0,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 brightness_range=None,
                 shear_range=0.0,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 fill_mode='nearest',
                 cval=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 interpolation_order=1,
                 dtype='float32',
                 directory=None,
                 x_col="filename",
                 y_col="class",
                 weight_col=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 validate_filenames=True,
                 fit_sample_size=None,
                 **kwargs):
        super(ImageDataFrameBatchGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening, zca_epsilon=zca_epsilon,
            rotation_range=rotation_range, width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range, zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode, cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip, rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=0.0, dtype=dtype)

        self.dataframe = dataframe
        self.directory = directory or ''
        self.x_col = x_col
        self.y_col = y_col
        self.weight_col = weight_col
        self.target_size = target_size
        self.color_mode = color_mode
        self.classes = classes
        self.batch_sieze = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.subset = subset
        self.interpolation = interpolation
        self.validate_filenames = validate_filenames
        self.fit_sample_size = fit_sample_size or 1000
        self.kwargs = kwargs

    def _fit(self, X, augment=False, rounds=1, seed=None):
        """To replace the method `fit`.
        """
        self.fit(X, augment=augment, rounds=rounds, seed=seed)
        self._fit = True

    def flow(self, X, y=None, batch_size=None):
        df = self.dataframe.iloc[np.squeeze(X), :]

        if self.featurewise_center and not hasattr(self, '_fit'):
            X_sample = self.sample(X, sample_size=self.fit_sample_size,
                                   standardize=False)
            # TODO: support other fit parameters.
            self._fit(X_sample)

        return DataFrameIterator(
            df,
            self,
            directory=self.directory,
            x_col=self.x_col,
            y_col=self.y_col,
            weight_col=self.weight_col,
            target_size=self.target_size,
            color_mode=self.color_mode,
            classes=self.classes,
            class_mode=self.class_mode,
            data_format=self.data_format,
            batch_size=batch_size or self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
            save_to_dir=self.save_to_dir,
            save_prefix=self.save_prefix,
            save_format=self.save_format,
            interpolation=self.interpolation)

    def sample(self, X, sample_size=None, standardize=True):
        pass


# Refer to `DataFrameIterator._check_params`
def _check_params(df, x_col, y_col, weight_col, classes, class_mode):
    # check class mode is one of the currently supported
    if class_mode not in allowed_class_modes:
        raise ValueError('Invalid class_mode: {}; expected one of: {}'
                         .format(class_mode, allowed_class_modes))
    # check that y_col has several column names if class_mode is multi_output
    if (class_mode == 'multi_output') and not isinstance(y_col, list):
        raise TypeError(
            'If class_mode="{}", y_col must be a list. Received {}.'
            .format(class_mode, type(y_col).__name__)
        )
    # check that filenames/filepaths column values are all strings
    if not all(df[x_col].apply(lambda x: isinstance(x, str))):
        raise TypeError('All values in column x_col={} must be strings.'
                        .format(x_col))
    # check labels are string if class_mode is binary or sparse
    if class_mode in {'binary', 'sparse'}:
        if not all(df[y_col].apply(lambda x: isinstance(x, str))):
            raise TypeError('If class_mode="{}", y_col="{}" column '
                            'values must be strings.'
                            .format(class_mode, y_col))
    # check that if binary there are only 2 different classes
    if class_mode == 'binary':
        if classes:
            classes = set(classes)
            if len(classes) != 2:
                raise ValueError('If class_mode="binary" there must be 2 '
                                 'classes. {} class/es were given.'
                                 .format(len(classes)))
        elif df[y_col].nunique() != 2:
            raise ValueError('If class_mode="binary" there must be 2 classes. '
                             'Found {} classes.'.format(df[y_col].nunique()))
    # check values are string, list or tuple if class_mode is categorical
    if class_mode == 'categorical':
        types = (str, list, tuple)
        if not all(df[y_col].apply(lambda x: isinstance(x, types))):
            raise TypeError('If class_mode="{}", y_col="{}" column '
                            'values must be type string, list or tuple.'
                            .format(class_mode, y_col))
    # raise warning if classes are given but will be unused
    if classes and class_mode in {"input", "multi_output", "raw", None}:
        warnings.warn('`classes` will be ignored given the class_mode="{}"'
                      .format(class_mode))
    # check that if weight column that the values are numerical
    if weight_col and not issubclass(df[weight_col].dtype.type, np.number):
        raise TypeError('Column weight_col={} must be numeric.'
                        .format(weight_col))


_filter_classes = DataFrameIterator._filter_classes


# refer to `DataFrameIterator._filter_valid_filepaths`
def _filter_valid_filepaths(df, x_col, directory):
    """Keep only dataframe rows with valid filenames

    # Arguments
        df: Pandas dataframe containing filenames in a column
        x_col: string, column in `df` that contains the filenames or filepaths

    # Returns
        absolute paths to image files
    """
    filepaths = df[x_col].map(
        lambda fname: os.path.join(directory, fname)
    )
    mask = filepaths.apply(validate_filename, args=(white_list_formats,))
    n_invalid = (~mask).sum()
    if n_invalid:
        warnings.warn(
            'Found {} invalid image filename(s) in x_col="{}". '
            'These filename(s) will be ignored.'
            .format(n_invalid, x_col)
        )
    return df[mask]
