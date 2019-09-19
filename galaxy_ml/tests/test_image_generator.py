import pandas as pd
import numpy as np

from galaxy_ml.model_validations import train_test_split
from galaxy_ml.preprocessors import ImageDataFrameBatchGenerator
from galaxy_ml.preprocessors._image_batch_generator import \
    clean_image_dataframe
from sklearn.base import clone


train_test_split.__test__ = False


def test_image_dataframe_generator():
    df = pd.read_csv('./tools/test-data/trainLabels_500.tsv', sep='\t')
    df['id'] = df['id'].apply(lambda x: str(x) + '.png')
    directory = './tools/test-data/cifar-10_500/'

    x_col = 'id'
    y_col = 'label'
    class_mode = 'categorical'

    df = clean_image_dataframe(df, directory, x_col=x_col, y_col=y_col,
                               class_mode=class_mode)

    image_generator = ImageDataFrameBatchGenerator(
        df, directory=directory,
        x_col=x_col, y_col=y_col,
        shuffle=False, seed=42,
        class_mode=class_mode,
        target_size=(32, 32),
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fit_sample_size=None)

    image_generator2 = clone(image_generator)
    image_generator2.set_processing_attrs()

    X = np.arange(df.shape[0])[:, np.newaxis]
    y = image_generator2.labels

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42,
                                       shuffle='stratified', labels=y)

    image_gen = image_generator2.flow(X_train, batch_size=32)

    batch_X, batch_y = next(image_gen)

    assert image_generator2.class_indices == \
        {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
         'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    assert image_generator2.classes[:10] == [6, 9, 9, 4, 1, 1, 2, 7, 8, 3], \
        image_generator2.classes[:10]

    assert np.squeeze(X_train)[:10].tolist() == [7, 280, 317, 299, 433, 401,
                                                 171, 155,  82, 197], \
        np.squeeze(X_train)[:10].tolist()

    assert batch_X.shape == (32, 32, 32, 3), batch_X.shape
    assert np.argmax(batch_y[:10], axis=1).tolist() == \
        [7, 8, 0, 4, 7, 0, 2, 8, 4, 3], np.argmax(batch_y[:10], axis=1)
