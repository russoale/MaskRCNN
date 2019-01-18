import warnings
import numpy as np
from keras import optimizers
from keras import callbacks
from keras import backend as K

from mrcnn.utility_functions import log


def save_all_weights(model, filepath, layers=None,
                     include_optimizer=True):
    """
    Save model weights and optimizer weights but not configuration to a HDF5 file.
    Functionally between `save` and `save_weights`.
    The HDF5 file contains:
        - the model's weights
        - the model's optimizer's state (if any)
    If you have a complicated model or set of models that do not serialize to JSON correctly, use this method.
    # Arguments
        model: Keras model instance to be saved.
        filepath: String, path where to save the model.
        layers: (Optional) The layers to save the weights from.
             Defaults to model.layers
        include_optimizer: If True, save optimizer's state together.
    # Raises
        ImportError: if h5py is not available.
    """
    import h5py
    # Conditional import to support versions of Keras before 2.2
    # TODO: remove in about 6 months (end of 2018)
    try:
        from keras.engine import saving
    except ImportError:
        # Keras before 2.2 used the 'topology' namespace.
        from keras.engine import topology as saving
    if h5py is None:
        raise ImportError('`save_all_weights` requires h5py.')

    with h5py.File(filepath, 'w') as f:
        model_weights_group = f.create_group('model_weights')
        if layers is not None:
            model_layers = layers
        else:
            model_layers = model.layers
        saving.save_weights_to_hdf5_group(model_weights_group, model_layers)

        if include_optimizer and hasattr(model, 'optimizer') and model.optimizer:
            if isinstance(model.optimizer, optimizers.TFOptimizer):
                warnings.warn(
                    'TensorFlow optimizers do not '
                    'make it possible to access '
                    'optimizer attributes or optimizer state '
                    'after instantiation. '
                    'As a result, we cannot save the optimizer '
                    'as part of the model save file.'
                    'You will have to compile your model again after loading it. '
                    'Prefer using a Keras optimizer instead '
                    '(see keras.io/optimizers).')
            else:
                # Save optimizer weights.
                symbolic_weights = getattr(model.optimizer, 'weights')
                if symbolic_weights:
                    optimizer_weights_group = f.create_group('optimizer_weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    weight_names = []
                    for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                        # Default values of symbolic_weights is /variable for theano
                        if K.backend() == 'theano':
                            if hasattr(w, 'name') and w.name != "/variable":
                                name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        else:
                            if hasattr(w, 'name') and w.name:
                                name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        weight_names.append(name.encode('utf8'))
                    optimizer_weights_group.attrs['weight_names'] = weight_names
                    log("Saving optimizer weights as well")
                    for name, val in zip(weight_names, weight_values):
                        param_dset = optimizer_weights_group.create_dataset(
                            name,
                            val.shape,
                            dtype=val.dtype)
                        if not val.shape:
                            # scalar
                            param_dset[()] = val
                        else:
                            param_dset[:] = val


def load_weights(model, filepath, by_name=False, exclude=None, include_optimizer=True):
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    include_optimizer: Load optimzer weights as well. Model has to be compiled beforehand.
    """

    if exclude:
        by_name = True

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    keras_model = model.keras_model
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
        else keras_model.layers

    if include_optimizer:
        # Necessary to initialize the weight variables of the optimizer
        keras_model._make_train_function()
        assert keras_model.optimizer, "Model needs to be compiled with an optimizer to load optimizer weights"

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)

    load_all_weights(keras_model, filepath, layers=layers, by_name=by_name,
                     include_optimizer=include_optimizer)

    # Update the log directory
    model.set_log_dir(filepath)


def load_all_weights(model, filepath, layers=None, by_name=False,
                     include_optimizer=True):
    """Loads the weights of a model saved via `save_all_weights`.
    If model has been compiled, optionally load its optimizer's weights.
    # Arguments
        model: instantiated model with architecture matching the saved model.
            Compile the model beforehand if you want to load optimizer weights.
        filepath: String, path to the saved model.
        layers: (Optional) The layers to load the weights into.
             Defaults to model.layers
        by_name: (Optional)
    # Returns
        None. The model will have its weights updated.
    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import h5py
    from keras.engine import saving

    with h5py.File(filepath, mode='r') as loaded_weights:
        # set loaded weights
        if 'layer_names' not in loaded_weights.attrs and 'model_weights' in loaded_weights:
            model_f = loaded_weights['model_weights']
        else:
            model_f = loaded_weights

        # only loading specified layers if available
        if layers is None:
            model_layers = model.layers
        else:
            model_layers = layers

        # load weight by layer name if set to true
        if by_name:
            saving.load_weights_from_hdf5_group_by_name(model_f, model_layers)
        else:
            saving.load_weights_from_hdf5_group(model_f, model_layers)

        # set optimizer weights.
        if include_optimizer:
            if 'optimizer_weights' in loaded_weights and hasattr(model, 'optimizer') and model.optimizer:
                log("Loading optimizer weights as well")
                optimizer_weights_group = loaded_weights['optimizer_weights']
                optimizer_weight_names = [n.decode('utf8') for n in
                                          optimizer_weights_group.attrs['weight_names']]
                optimizer_weight_values = [optimizer_weights_group[n] for n in
                                           optimizer_weight_names]
                model.optimizer.set_weights(optimizer_weight_values)
            else:
                warnings.warn("Optimizer weights not included in weight file!", RuntimeWarning)


class ModelCheckpointWithOptimizer(callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        save_optimizer_weights: if True, then the optimizer weights will be saved
            as well.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 save_optimizer_weights=True,
                 mode='auto', period=1):
        super(ModelCheckpointWithOptimizer, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_optimizer_weights = save_optimizer_weights
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            if self.save_optimizer_weights:
                                layers = self.model.inner_model.layers if hasattr(self.model, "inner_model") \
                                    else self.model.layers
                                save_all_weights(self.model, filepath, layers=layers,
                                                 include_optimizer=True)
                            else:
                                self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    if self.save_optimizer_weights:
                        layers = self.model.inner_model.layers if hasattr(self.model, "inner_model") \
                            else self.model.layers
                        save_all_weights(self.model, filepath, layers=layers,
                                         include_optimizer=True)
                    else:
                        self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
