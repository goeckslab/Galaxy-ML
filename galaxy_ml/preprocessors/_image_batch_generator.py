"""
ImageDataFrameBatchGenerator + ImageFilesIterator vs.
ImageDataGenerator + DataFrameIterator in Keras

1) Both generate image batch tensor data from a `pandas.DataFrame` file
    with on-line augumentation.

But,

2) dataframe spit has been stripped, as we plan to use sklearn splitters
    to manage the validation splitting.
3) dataframe consistance check and cleanup are moved out into utility
    function `clean_image_dataframe`, which is better to be used before
    data splitting and aslo before passing to a generator instance.
4) all parameters are initiated in the `ImageDataFrameBatchGenerator` and
    `ImageFilesIterator` becomes very simplified.
5) `ImageDataFrameBatchGenerator` is sklearn api - compatible.
"""

import os
import warnings

from keras.utils import Sequence

from keras_preprocessing.image import (
    DataFrameIterator,
    ImageDataGenerator,
    Iterator
)
from keras_preprocessing.image.utils import (
    array_to_img,
    img_to_array,
    load_img,
    validate_filename,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


allowed_class_modes = {
    'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse', None}

white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')


class ImageFilesIterator(Iterator, BaseEstimator, Sequence):
    """ Override `keras_preprocessing.image.DataFrameIterator`

    Parameters
    -----------
    X : 2D-array
        Index array
    image_data_generator : Instance of ImageDataFrameBatchGenerator
    batch_size : Int. Default is 32
    """
    def __init__(self, X,
                 image_data_generator,
                 batch_size=32):
        self.X = np.squeeze(X)
        self.image_data_generator = image_data_generator
        self.filepaths = self.image_data_generator.filepaths
        self.sample_weight = self.image_data_generator.sample_weight
        self.labels = self.image_data_generator.labels
        self.target_size = self.image_data_generator.target_size
        self.color_mode = self.image_data_generator.color_mode
        self.data_format = self.image_data_generator.data_format
        self.image_shape = self.image_data_generator.image_shape
        self.save_to_dir = self.image_data_generator.save_to_dir
        self.save_prefix = self.image_data_generator.save_prefix
        self.save_format = self.image_data_generator.save_format
        self.interpolation = self.image_data_generator.interpolation
        self.class_mode = self.image_data_generator.class_mode
        self.dtype = self.image_data_generator.dtype
        self.shuffle = self.image_data_generator.shuffle
        self.seed = self.image_data_generator.seed
        self.class_indices = getattr(self.image_data_generator,
                                     'class_indices', None)
        self.classes = getattr(self.image_data_generator,
                               'classes', None)

        super(ImageFilesIterator, self).__init__(
            len(self.X), batch_size, self.shuffle, self.seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        index_array = self.X[index_array]
        batch_x = np.zeros((len(index_array),) + self.image_shape,
                           dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(
                    x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]


class ImageDataFrameBatchGenerator(ImageDataGenerator, BaseEstimator):
    """ Extend `keras_preprocessing.image.ImageDataGenerator` to work with
    DataFrame exclusively, generating batches of tensor data from
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
    fill_mode : One of {"constant", "nearest", "reflect" or "wrap"}.
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
    dataframe : Pandas dataframe containing the filepaths relative to
        `directory`. From `keras_preprocessing.image.ImageDataGenerator.
        flow_from_dataframe`.
    directory : string, path to the directory to read images from. If `None`,
        data in `x_col` column should be absolute paths.
    x_col : string, column in `dataframe` that contains the filenames (or
                absolute paths if `directory` is `None`).
    y_col : string or list, column/s in `dataframe` that has the target data.
    weight_col : string, column in `dataframe` that contains the sample
        weights. Default: `None`.
    target_size : tuple of integers `(height, width)`, default: `(256, 256)`.
        The dimensions to which all images found will be resized.
    color_mode : one of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to have 1 or 3 color channels.
    classes : optional list of classes (e.g. `['dogs', 'cats']`).
        Default: None. If None, all classes in `y_col` will be used.
    class_mode : one of "binary", "categorical", "input", "multi_output",
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
    shuffle : whether to shuffle the data (default: True)
    seed : optional random seed for shuffling and transformations.
    save_to_dir : Optional directory where to save the pictures
        being yielded, in a viewable format. This is useful
        for visualizing the random transformations being
        applied, for debugging purposes.
    save_prefix : String prefix to use for saving sample
        images (if `save_to_dir` is set).
    save_format : Format to use for saving sample images
        (if `save_to_dir` is set).
    interpolation : Interpolation method used to resample the image if the
        target size is different from that of the loaded image.
        Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
        If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
        supported. If PIL version 3.4.0 or newer is installed, `"box"` and
        `"hamming"` are also supported. By default, `"nearest"` is used.
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
                 shuffle=True,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 interpolation='nearest',
                 fit_sample_size=None,
                 **kwargs):
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.interpolation_order = interpolation_order
        self.dtype = dtype
        self.dataframe = dataframe
        self.directory = directory or ''
        self.x_col = x_col
        self.y_col = y_col
        self.weight_col = weight_col
        self.target_size = target_size
        self.classes = classes
        self.class_mode = class_mode
        self.shuffle = shuffle
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.fit_sample_size = fit_sample_size
        self.kwargs = kwargs

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
        self.data_format = data_format

        if not (np.isscalar(zoom_range)
                or len(zoom_range) == 2):
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))
        self.zoom_range = zoom_range

        if brightness_range is not None:
            if (
                not isinstance(brightness_range, (tuple, list))
                or len(brightness_range) != 2
            ):
                raise ValueError(
                    '`brightness_range should be tuple or list of two floats. '
                    'Received: %s' % (brightness_range,))
        self.brightness_range = brightness_range

        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        if self.class_mode in {"multi_output", "raw"}:
            return self._targets
        else:
            return self.classes

    @property
    def sample_weight(self):
        return self._sample_weight

    def set_processing_attrs(self):
        """set more `ImageDataGenerator` attributes
        """
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if self.data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        if np.isscalar(self.zoom_range):
            self.zoom_range = [1 - self.zoom_range, 1 + self.zoom_range]
        else:
            self.zoom_range = [self.zoom_range[0], self.zoom_range[1]]

        if self.zca_whitening:
            if not self.featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if self.featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if self.featurewise_std_normalization:
            if not self.featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if self.samplewise_std_normalization:
            if not self.samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')

        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        if self.class_mode not in ["input", "multi_output", "raw", None]:
            _, classes = DataFrameIterator._filter_classes(self.dataframe,
                                                           self.y_col,
                                                           self.classes)
            num_classes = len(classes)
            # build an index of all the unique classes
            self.class_indices = dict(zip(classes, range(len(classes))))
            self.classes = self.get_classes(self.dataframe, self.y_col)

        if self.class_mode == "multi_output":
            self._targets = [np.array(self.dataframe[col].tolist())
                             for col in self.y_col]
        if self.class_mode == "raw":
            self._targets = self.dataframe[self.y_col].values

        self.filenames = list(map(str, self.dataframe[self.x_col]))
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]

        self._sample_weight = self.dataframe[self.weight_col].values \
            if self.weight_col else None

        self.rng_ = check_random_state(self.seed)

        if self.class_mode in ["input", "multi_output", "raw", None]:
            print('Found {} image filenames.'
                  .format(len(self.filenames)))
        else:
            print('Found {} image filenames belonging to {} classes.'
                  .format(len(self.filenames), num_classes))

    def flow(self, X, y=None, batch_size=None, sample_weight=None):
        """ Return a Sequence iterator
        """
        if not hasattr(self, 'channel_axis'):
            self.set_processing_attrs()

        if self.featurewise_center and not hasattr(self, 'mean'):
            X_sample = self.sample(X, sample_size=self.fit_sample_size,
                                   standardize=False)
            if isinstance(X_sample, tuple):
                X_sample = X_sample[0]
            # TODO: support other fit parameters.
            self.fit(X_sample)

        return ImageFilesIterator(X, self, batch_size=batch_size)

    def sample(self, X=None, sample_size=None, standardize=True):
        """ Retrived fix-sized image tersors

        Parameters
        ----------
        X : 2D-array. Default is None
            Expanded sub-index array of the dataframe.
            If None, X = np.arange(n_samples)[:, np.newaxis].
        sample_size : int. Default is None.
            The number of samples to be retrieved.
            If None, sample_size = X.shape[0]
        standardize : bool. Default is True.
            Whether to transform the image tersor data.
            If False, return direct results of `img_to_array`.
        """
        if X is None:
            X = np.arange(self.dataframe.shape[0])[:, np.newaxis]
        if not sample_size:
            sample_size = X.shape[0]

        retrieved_X = np.zeros((sample_size,) + self.image_shape,
                               dtype=self.dtype)

        filepaths = self.filepaths
        indices = np.squeeze(X)
        sample_index = self.rng_.choice(indices, size=sample_size,
                                        replace=False)
        for i, j in enumerate(sample_index):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)

            x = img_to_array(img, data_format=self.data_format)
            if hasattr(img, 'close'):
                img.close()

            if not standardize:
                retrieved_X[i] = x
                continue
            params = self.get_random_transform(x.shape)
            x = self.apply_transform(x, params)
            x = self.standardize(x)
            retrieved_X[i] = x

        if self.save_to_dir:
            for i, j in enumerate(sample_index):
                img = array_to_img(retrieved_X[i], self.data_format,
                                   scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # retrieve labels
        if self.class_mode == 'input':
            retrieved_y = retrieved_X.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            retrieved_y = np.empty(sample_size, dtype=self.dtype)
            for i, n_observation in enumerate(sample_index):
                retrieved_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            retrieved_y = np.zeros((sample_size, len(self.class_indices)),
                                   dtype=self.dtype)
            for i, n_observation in enumerate(sample_index):
                retrieved_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            retrieved_y = [output[sample_index] for output in self.labels]
        elif self.class_mode == 'raw':
            retrieved_y = self.labels[sample_index]
        else:
            return retrieved_X

        return retrieved_X, retrieved_y

    def get_classes(self, df, y_col):
        labels = []
        for label in df[y_col]:
            if isinstance(label, (list, tuple)):
                labels.append([self.class_indices[lbl] for lbl in label])
            else:
                labels.append(self.class_indices[label])
        return labels


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


def clean_image_dataframe(df, directory=None, x_col='filename',
                          y_col='class', weight_col=None,
                          classes=None, class_mode='categorical',
                          drop_duplicates=True,
                          validate_filenames=True):
    """utils to check and clean up the dataframe containing image info.
    Be used before train_test splitting and before passing to a
    `ImageDataFrameGenerator`.

    Parameters
    ----------
    df : pandas.DataFrame object.
        Contains image file paths, classes and so on.
    directory : str. Default is None.
        A common folder prefix.
    x_col : str. Default = 'filename'.
        A column name in `df`. The pointed column contains image file
        names or relative paths to the `directory`.
    y_col : str. Default = 'class'.
        Column name(s) in `df`. The pointed column(s) contain class labels.
    weight_col : str. Default is None.
        A column name is `df`. The pointed column contains sample weights.
    classes : list or tuple. Default is None.
        A set of class labels to be predicted.
    class_mode : str or None. Default = 'categorical'.
        One of the ['binary', 'categorical', 'input', 'multi_output',
        'raw', 'sparse', None].
    drop_duplicates : bool. Default is True.
        Whether to drop duplicates in the dataframe.
    validate_filenames : bool. Default is True.
        Whether to filter valid file paths.
    """
    if drop_duplicates:
        df.drop_duplicates(inplace=True)
    _check_params(df, x_col, y_col, weight_col, classes, class_mode)
    if validate_filenames:
        df = _filter_valid_filepaths(df, x_col, directory)
    # this seems to be repeat computation
    # TODO remove it
    if class_mode not in ["input", "multi_output", "raw", None]:
        df, _ = DataFrameIterator._filter_classes(df, y_col, classes)

    return df
