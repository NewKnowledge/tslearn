"""
The :mod:`tslearn.shapelets` module gathers Shapelet-based algorithms.

It depends on the `keras` library for optimization.
"""

from keras.models import Model
from keras.layers import Dense, Conv1D, Layer, Input, concatenate, add
from keras.metrics import categorical_accuracy, categorical_crossentropy, binary_accuracy, binary_crossentropy
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.regularizers import l2
from keras.initializers import Initializer
import keras.backend as K
from keras.engine import InputSpec
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, TensorBoard, TerminateOnNaN, EarlyStopping)
import os
from datetime import datetime
import time
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

import numpy
from tensorflow import set_random_seed
from keras.utils import Sequence
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class GlobalMinPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def __init__(self, **kwargs):
        super(GlobalMinPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.min(inputs, axis=1)


class GlobalArgminPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def __init__(self, **kwargs):
        super(GlobalArgminPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.cast(K.argmin(inputs, axis=1), dtype=K.floatx())


def _kmeans_init_shapelets(X, n_shapelets, shp_len, n_draw=10000):
    n_ts, sz, d = X.shape
    indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
    indices_time = numpy.random.choice(sz - shp_len + 1, size=n_draw, replace=True)
    subseries = numpy.zeros((n_draw, shp_len, d))
    for i in range(n_draw):
        subseries[i] = X[indices_ts[i], indices_time[i]:indices_time[i] + shp_len]
    return TimeSeriesKMeans(n_clusters=n_shapelets,
                            metric="euclidean",
                            verbose=False).fit(subseries).cluster_centers_


class KMeansShapeletInitializer(Initializer):
    """Initializer that generates shapelet tensors based on a clustering of time series snippets.
    # Arguments
        dataset: a dataset of time series.
    """
    def __init__(self, X):
        self.X_ = to_time_series_dataset(X)

    def __call__(self, shape, dtype=None):
        n_shapelets, shp_len = shape
        shapelets = _kmeans_init_shapelets(self.X_,
                                           n_shapelets,
                                           shp_len)[:, :, 0]
        return K.tensorflow_backend._to_tensor(x=shapelets, dtype=K.floatx())

    def get_config(self):
        return {'data': self.X_}


class LocalSquaredDistanceLayer(Layer):
    """Pairwise (squared) distance computation between local patches and shapelets
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        3D tensor with shape:
        `(batch_size, steps, n_shapelets)`
    """
    def __init__(self, n_shapelets, X=None, **kwargs):
        self.n_shapelets = n_shapelets
        if X is None or K.backend() != "tensorflow":
            self.initializer = "uniform"
        else:
            self.initializer = KMeansShapeletInitializer(X)
        super(LocalSquaredDistanceLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_shapelets, input_shape[2]),
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalSquaredDistanceLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        # (x - y)^2 = x^2 + y^2 - 2 * x * y
        x_sq = K.expand_dims(K.sum(x ** 2, axis=2), axis=-1)
        y_sq = K.reshape(K.sum(self.kernel ** 2, axis=1), (1, 1, self.n_shapelets))
        xy = K.dot(x, K.transpose(self.kernel))
        return (x_sq + y_sq - 2 * xy) / K.int_shape(self.kernel)[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.n_shapelets


def grabocka_params_to_shapelet_size_dict(n_ts, ts_sz, n_classes, l, r):
    """Compute number and length of shapelets.

     This function uses the heuristic from [1]_.

    Parameters
    ----------
    n_ts: int
        Number of time series in the dataset
    ts_sz: int
        Length of time series in the dataset
    n_classes: int
        Number of classes in the dataset
    l: float
        Fraction of the length of time series to be used for base shapelet length
    r: int
        Number of different shapelet lengths to use

    Returns
    -------
    dict
        Dictionnary giving, for each shapelet length, the number of such shapelets to be generated

    Examples
    --------
    >>> d = grabocka_params_to_shapelet_size_dict(n_ts=100, ts_sz=100, n_classes=3, l=0.1, r=2)
    >>> keys = sorted(d.keys())
    >>> print(keys)
    [10, 20]
    >>> print([d[k] for k in keys])
    [4, 4]


    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    base_size = int(l * ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        n_shapelets = int(numpy.log10(n_ts * (ts_sz - shp_sz + 1) * (n_classes - 1)))
        d[shp_sz] = n_shapelets
    return d


class ShapeletModel(BaseEstimator, ClassifierMixin):
    """Learning Time-Series Shapelets model.


    Learning Time-Series Shapelets was originally presented in [1]_.

    Parameters
    ----------
    n_shapelets_per_size: dict
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
    max_iter: int (default: 1000)
        Number of training epochs.
    batch_size: int (default:256)
        Batch size to be used.
    verbose_level: {0, 1, 2} (default: 2)
        `keras` verbose level.
    optimizer: str or keras.optimizers.Optimizer (default: "sgd")
        `keras` optimizer to use for training.
    weight_regularizer: float or None (default: None)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer.
        If None, no regularization is performed.
    random_state : int or None, optional (default: None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    shapelets_: numpy.ndarray of objects, each object being a time series
        Set of time-series shapelets.
    shapelets_as_time_series_: numpy.ndarray of shape (n_shapelets, sz_shp, d) where \
    sz_shp is the maximum of all shapelet sizes
        Set of time-series shapelets formatted as a ``tslearn`` time series dataset.

    Note
    ----
        This implementation requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, d=2, n_blobs=3)
    >>> clf = ShapeletModel(n_shapelets_per_size={10: 5}, max_iter=1, verbose_level=0)
    >>> clf.fit(X, y).shapelets_.shape
    (5,)
    >>> clf.shapelets_[0].shape
    (10, 2)
    >>> clf.predict(X).shape
    (60,)
    >>> clf.predict_proba(X).shape
    (60, 3)
    >>> clf.transform(X).shape
    (60, 5)
    >>> params = clf.get_params(deep=True)
    >>> sorted(params.keys())
    ['batch_size', 'max_iter', 'n_shapelets_per_size', 'optimizer', 'random_state', 'verbose_level', 'weight_regularizer']
    >>> clf.set_params(batch_size=128)  # doctest: +NORMALIZE_WHITESPACE
    ShapeletModel(batch_size=128, max_iter=1, n_shapelets_per_size={10: 5},
           optimizer='sgd', random_state=None, verbose_level=0,
           weight_regularizer=0.0)
    >>> clf2 = ShapeletModel(n_shapelets_per_size={10: 5, 20: 10}, max_iter=1, verbose_level=0)
    >>> clf2.fit(X, y).shapelets_.shape
    (15,)
    >>> clf2.shapelets_[0].shape
    (10, 2)
    >>> clf2.shapelets_[5].shape
    (20, 2)
    >>> clf2.shapelets_as_time_series_.shape
    (15, 20, 2)
    >>> clf2.predict(X).shape
    (60,)
    >>> clf2.transform(X).shape
    (60, 15)
    >>> clf2.locate(X).shape
    (60, 15)
    >>> import sklearn
    >>> cv_results = sklearn.model_selection.cross_validate(clf, X, y, return_train_score=False)
    >>> cv_results['test_score'].shape
    (3,)

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self, n_shapelets_per_size,
                 max_iter=1000,
                 batch_size=256,
                 verbose_level=2,
                 optimizer="sgd",
                 weight_regularizer=0.,
                 random_state=None):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.n_classes = None
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.weight_regularizer = weight_regularizer
        self.random_state = random_state
        self.model = None
        self.transformer_model = None
        self.locator_model = None
        self.batch_size = batch_size
        self.verbose_level = verbose_level
        self.categorical_y = False
        self.label_binarizer = None
        self.binary_problem = False

        self.d = None

    @property
    def _n_shapelet_sizes(self):
        return len(self.n_shapelets_per_size)

    @property
    def shapelets_(self):
        total_n_shp = sum(self.n_shapelets_per_size.values())
        shapelets = numpy.empty((total_n_shp, ), dtype=object)
        idx = 0
        for i, shp_sz in enumerate(sorted(self.n_shapelets_per_size.keys())):
            n_shp = self.n_shapelets_per_size[shp_sz]
            for idx_shp in range(idx, idx + n_shp):
                shapelets[idx_shp] = numpy.zeros((shp_sz, self.d))
            for di in range(self.d):
                for inc, shp in enumerate(self.model.get_layer("shapelets_%d_%d" % (i, di)).get_weights()[0]):
                    shapelets[idx + inc][:, di] = shp
            idx += n_shp
        assert idx == total_n_shp
        return shapelets

    @property
    def shapelets_as_time_series_(self):
        total_n_shp = sum(self.n_shapelets_per_size.values())
        shp_sz = max(self.n_shapelets_per_size.keys())
        non_formatted_shapelets = self.shapelets_
        d = non_formatted_shapelets[0].shape[1]
        shapelets = numpy.zeros((total_n_shp, shp_sz, d)) + numpy.nan
        for i in range(self._n_shapelet_sizes):
            sz = non_formatted_shapelets[i].shape[0]
            shapelets[i, :sz, :] = non_formatted_shapelets[i]
        return shapelets

    def clear_session(self):
        K.clear_session()
        return

    def _get_timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    def _get_checkpoint_path(self, source_dir, timestamp):
        data_dir, data_fname = os.path.split(source_dir)
        data_fname += timestamp
        return os.path.join(data_dir, 'checkpoints', data_fname + '_{epoch:02d}-{val_loss:.4f}.h5')

    def _get_callbacks(self, source_dir, batch_size, chkpt_period = 1):
        terminate_on_nan = TerminateOnNaN()

        timestamp = self._get_timestamp()
        tb_logdir = 'logs/{0}'.format(timestamp)
        tensorboard = TensorBoard(tb_logdir, histogram_freq = 0, batch_size = batch_size)
        reducelr = ReduceLROnPlateau(factor = 0.2, patience = 3)
        if source_dir is not None:
            checkpoint_path = self._get_checkpoint_path(source_dir, timestamp)
            chkpt_dir = os.path.split(checkpoint_path)[0]
            if not os.path.isdir(chkpt_dir):
                os.makedirs(chkpt_dir)
            checkpointer = ModelCheckpoint(checkpoint_path, period=chkpt_period, monitor='val_loss', save_best_only=True, mode = 'min')
        earlystopping = EarlyStopping(monitor = 'val_loss', patience =10)

        return [terminate_on_nan, reducelr, checkpointer, earlystopping] # add in tensorboard for real training runs

    def _fit_helper(self, X, y):
        """Learn time-series shapelets.
        Helper fit function that supports fit and fit_generator

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y : array-like of shape=(n_ts, )
            Time series labels.
        """
        set_random_seed(seed=self.random_state)
        numpy.random.seed(seed=self.random_state)
        n_ts, sz, d = X.shape
        self.d = d
        if y.ndim == 1:
            self.label_binarizer = LabelBinarizer().fit(y)
            y_ = self.label_binarizer.transform(y)
        else:
            y_ = y
            self.categorical_y = True
            assert y_.shape[1] != 2, "Binary classification case, monodimensional y should be passed."
        if y_.ndim == 1:
            n_classes = 2
        else:
            n_classes = y_.shape[1]
        
        return n_ts, sz, n_classes

    def generate_model(self, sz, n_classes):
        '''
            Generate structure of model used for Shapelet classifier
        '''
        self._set_model_layers(X=None, ts_sz=sz, d=self.d, n_classes=n_classes)
        return self.model

    def fit(self, X, y, source_dir = None, val_split = 0.3):
        """Learn time-series shapelets.
        Helper fit function that supports fit and fit_generator

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y : array-like of shape=(n_ts, )
            Time series labels.fit
        """
        n_ts, sz, n_classes = self._fit_helper(X, y)
        self._set_model_layers(X=X, ts_sz=sz, d=self.d, n_classes=n_classes)
        self.transformer_model.compile(loss="mean_squared_error",
                                       optimizer=self.optimizer)
        self.locator_model.compile(loss="mean_squared_error",
                                   optimizer=self.optimizer)
        self._set_weights_false_conv(d=self.d)

        callbacks = self._get_callbacks(source_dir, self.batch_size)
        history = self.model.fit([X[:,:,di].reshape((n_ts, sz, 1)) for di in range(self.d)], y, 
                       validation_split = val_split,
                       callbacks = callbacks,
                       epochs=self.max_iter,
                       batch_size = self.batch_size,
                       verbose=self.verbose_level,
                       shuffle = True)
        return self

    def fit_hp_opt(self, X, y, source_dir, val_split = 0.7, epochs = 100):
        """Learn time-series shapelets.
        Helper fit function that supports fit and fit_generator

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y : array-like of shape=(n_ts, )
            Time series labels.
        """
        n_ts, sz, n_classes = self._fit_helper(X, y)
        self._set_model_layers_hp_opt(X=X, ts_sz=sz, d=self.d, n_classes=n_classes)
        self.transformer_model.compile(loss="mean_squared_error",
                                       optimizer={{choice(['adam', 'adagrad', 'rmsprop', 'sgd'])}})
        self.locator_model.compile(loss="mean_squared_error",
                                   optimizer={{choice(['adam', 'adagrad', 'rmsprop', 'sgd'])}})
        self._set_weights_false_conv(d=self.d)

        # for HP tuning simply fit once, no generator
        callbacks = self._get_callbacks(source_dir, self.batch_size)
        
        self.model.fit([X_train[:,:,di].reshape((n_ts, sz, 1)) for di in range(self.d)], y_train,
                       batch_size = {{choice([64, 128, 256, 512])}},
                       epochs=epochs,
                       shuffle=True,
                       verbose=self.verbose_level, 
                       validation_split = val_split)
        _, acc = self.model.evaluate(X_val, y_val, show_accuracy = True, verbose = 0)
        best_run, best_model = optim.minimize(model = {'loss':-acc, 'status': STATUS_OK, 'model': self.model}, 
            data = (lambda: X_train, y_train, X_val, y_val), algo = tpe.suggest, max_evals = 10, trials = Trials())
        print('Best performing model chosen hyper-parameters:')
        print(best_run)
        return best_run, best_model

    def predict(self, X):
        """Predict class for a given set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, ) or (n_ts, n_classes), depending on the shape of the \
        label vector provided at training time.
            Index of the cluster each sample belongs to or class probability matrix, depending on
            what was provided at training time.
        """
        categorical_preds = self.predict_proba(X)
        if self.categorical_y:
            return categorical_preds
        else:
            if categorical_preds.shape[1] == 2:
                categorical_preds = categorical_preds[:, 0]
            return self.label_binarizer.inverse_transform(categorical_preds)

    def predict_proba(self, X):
        """Predict class probability for a given set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_classes),
            Class probability matrix.
        """
        X_ = to_time_series_dataset(X)
        n_ts, sz, d = X_.shape
        categorical_preds = self.model.predict([X_[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d)],
                                               batch_size=self.batch_size,
                                               verbose=self.verbose_level)
        return categorical_preds

    def transform(self, X):
        """Generate shapelet transform for a set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_shapelets)
            Shapelet-Transform of the provided time series.
        """
        X_ = to_time_series_dataset(X)
        n_ts, sz, d = X_.shape
        pred = self.transformer_model.predict([X_[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d)],
                                              batch_size=self.batch_size,
                                              verbose=self.verbose_level)
        return pred

    def locate(self, X):
        """Compute shapelet match location for a set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_shapelets)
            Location of the shapelet matches for the provided time series.
        """
        X_ = to_time_series_dataset(X)
        n_ts, sz, d = X_.shape
        locations = self.locator_model.predict([X_[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d)],
                                          batch_size=self.batch_size,
                                          verbose=self.verbose_level)
        return locations.astype(numpy.int)

    def _set_weights_false_conv(self, d):
        shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        for i, sz in enumerate(shapelet_sizes):
            for di in range(d):
                self.model.get_layer("false_conv_%d_%d" % (i, di)).set_weights([numpy.eye(sz).reshape((sz, 1, sz))])

    def _set_model_layers_helper(self, X, ts_sz, d, n_classes):

        inputs = [Input(shape=(ts_sz, 1), name="input_%d" % di) for di in range(d)]
        shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        pool_layers = []
        pool_layers_locations = []
        for i, sz in enumerate(sorted(shapelet_sizes)):
            transformer_layers = [Conv1D(filters=sz,
                                         kernel_size=sz,
                                         trainable=False,
                                         use_bias=False,
                                         name="false_conv_%d_%d" % (i, di))(inputs[di]) for di in range(d)]
            shapelet_layers = [LocalSquaredDistanceLayer(self.n_shapelets_per_size[sz],
                                                         X=X,
                                                         name="shapelets_%d_%d" % (i, di))(transformer_layers[di])
                               for di in range(d)]
            if d == 1:
                summed_shapelet_layer = shapelet_layers[0]
            else:
                summed_shapelet_layer = add(shapelet_layers)
            pool_layers.append(GlobalMinPooling1D(name="min_pooling_%d" % i)(summed_shapelet_layer))
            pool_layers_locations.append(GlobalArgminPooling1D(name="min_pooling_%d" % i)(summed_shapelet_layer))
        if len(shapelet_sizes) > 1:
            concatenated_features = concatenate(pool_layers)
            concatenated_locations = concatenate(pool_layers_locations)
        else:
            concatenated_features = pool_layers[0]
            concatenated_locations = pool_layers_locations[0]
        return inputs, concatenated_features, concatenated_locations

    def _set_model_layers_hp_opt(self, X, ts_sz, d, n_classes):
        '''
            Set model layers with HP tuning
                Tune:   number of layers, number of hidden units, activation function, optimizer, learning rate
                        batch size, epochs, dropout, regularization
        '''
        inputs, concatenated_features, concatenated_locations = self._set_model_layers_helper(X, ts_sz, d, n_classes)

        # try different L2 weight regularizers
        outputs = Dense(units=n_classes if n_classes > 2 else 1,
                        activation="softmax" if n_classes > 2 else "sigmoid",
                        kernel_regularizer=l2({{choice([.001, .01, .1, 1])}}),
                        name="classification")(concatenated_features)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.transformer_model = Model(inputs=inputs, outputs=concatenated_features)
        self.locator_model = Model(inputs=inputs, outputs=concatenated_locations)

        # try different optimizers first, then try different learning rates
        self.model.compile(loss="categorical_crossentropy" if n_classes > 2 else "binary_crossentropy",
                           optimizer={{choice(['adam', 'adagrad', 'rmsprop', 'sgd'])}},
                           metrics=[categorical_accuracy, categorical_crossentropy] if n_classes > 2
                           else [binary_accuracy, binary_crossentropy])

    def _set_model_layers(self, X, ts_sz, d, n_classes):
        inputs, concatenated_features, concatenated_locations = self._set_model_layers_helper(X, ts_sz, d, n_classes)
        outputs = Dense(units=n_classes if n_classes > 2 else 1,
                        activation="softmax" if n_classes > 2 else "sigmoid",
                        kernel_regularizer=l2(self.weight_regularizer) if self.weight_regularizer > 0 else None,
                        name="classification")(concatenated_features)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.transformer_model = Model(inputs=inputs, outputs=concatenated_features)
        self.locator_model = Model(inputs=inputs, outputs=concatenated_locations)

        # try different optimizers
        self.model.compile(loss="categorical_crossentropy" if n_classes > 2 else "binary_crossentropy",
                           optimizer=self.optimizer,
                           metrics=[categorical_accuracy, categorical_crossentropy] if n_classes > 2
                           else [binary_accuracy, binary_crossentropy,])

    def get_weights(self, layer_name=None):
        """Return model weights (or weights for a given layer if `layer_name` is provided).

        Parameters
        ----------
        layer_name: str or None (default: None)
            Name of the layer for which  weights should be returned.
            If None, all model weights are returned.
            Available layer names with weights are:
            - "shapelets_i_j" with i an integer for the shapelet id and j an integer for the dimension
            - "classification" for the final classification layer

        Returns
        -------
        list
            list of model (or layer) weights

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X, y = random_walk_blobs(n_ts_per_blob=100, sz=256, d=1, n_blobs=3)
        >>> clf = ShapeletModel(n_shapelets_per_size={10: 5}, max_iter=0, verbose_level=0)
        >>> clf.fit(X, y).get_weights("classification")[0].shape
        (5, 3)
        """
        if layer_name is None:
            return self.model.get_weights()
        else:
            return self.model.get_layer(layer_name).get_weights()


class SerializableShapeletModel(ShapeletModel):
    """Serializable variant of the Learning Time-Series Shapelets model.


    Learning Time-Series Shapelets was originally presented in [1]_.

    Parameters
    ----------
    n_shapelets_per_size: dict
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
    max_iter: int (default: 1000)
        Number of training epochs.
    batch_size: int (default:256)
        Batch size to be used.
    verbose_level: {0, 1, 2} (default: 2)
        `keras` verbose level.
    learning_rate: float (default: 0.01)
        Learning rate to be used for the SGD optimizer.
    weight_regularizer: float or None (default: None)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer.
        If None, no regularization is performed.
    random_state : int or None, optional (default: None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    shapelets_: numpy.ndarray of objects, each object being a time series
        Set of time-series shapelets.
    shapelets_as_time_series_: numpy.ndarray of shape (n_shapelets, sz_shp, d) where \
    sz_shp is the maximum of all shapelet sizes
        Set of time-series shapelets formatted as a ``tslearn`` time series dataset.

    Note
    ----
        This implementation requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, d=2, n_blobs=3)
    >>> clf = SerializableShapeletModel(n_shapelets_per_size={10: 5}, max_iter=1, verbose_level=0, learning_rate=0.01)
    >>> clf.fit(X, y).shapelets_.shape
    (5,)
    >>> clf.shapelets_[0].shape
    (10, 2)
    >>> clf.predict(X).shape
    (60,)
    >>> clf.predict_proba(X).shape
    (60, 3)
    >>> clf.transform(X).shape
    (60, 5)
    >>> params = clf.get_params(deep=True)
    >>> sorted(params.keys())
    ['batch_size', 'learning_rate', 'max_iter', 'n_shapelets_per_size', 'random_state', 'verbose_level', 'weight_regularizer']
    >>> clf.set_params(batch_size=128)  # doctest: +NORMALIZE_WHITESPACE
    SerializableShapeletModel(batch_size=128, learning_rate=0.01, max_iter=1,
           n_shapelets_per_size={10: 5}, random_state=None,
           verbose_level=0, weight_regularizer=0.0)
    >>> import sklearn
    >>> cv_results = sklearn.model_selection.cross_validate(clf, X, y, return_train_score=False)
    >>> cv_results['test_score'].shape
    (3,)

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self, n_shapelets_per_size,
                 max_iter=1000,
                 batch_size=256,
                 verbose_level=2,
                 learning_rate=0.01,
                 weight_regularizer=0.,
                 random_state=None):
        super(SerializableShapeletModel,
              self).__init__(n_shapelets_per_size=n_shapelets_per_size,
                             max_iter=max_iter,
                             batch_size=batch_size,
                             verbose_level=verbose_level,
                             weight_regularizer=weight_regularizer,
                             random_state=random_state)
        self.learning_rate = learning_rate

    def _set_model_layers(self, X, ts_sz, d, n_classes):
        super(SerializableShapeletModel,
              self)._set_model_layers(X=X,
                                      ts_sz=ts_sz,
                                      d=d,
                                      n_classes=n_classes)
        K.set_value(self.model.optimizer.lr, 0.001)

    def set_params(self, **params):
        return super(SerializableShapeletModel, self).set_params(**params)

class ShapeletSequence(Sequence):
    ''' Generates a sequence of time series data
        
        x_set: set of input time series data
        y_set: set of output time series label
    '''

    def __init__(self, x_set, y_set, batch_size=256):
        self.batch_size = batch_size
        self.x = x_set
        self.y = y_set
        print('x shape in sequence gen {}'.format(self.x.shape))

    def __len__(self):
        return int(numpy.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return numpy.array([batch_x[:, :, di].reshape((self.batch_x.shape[0], self.batch_x.shape[1], 1)) for di in range(self.x.shape[2])]), numpy.array(batch_y)
