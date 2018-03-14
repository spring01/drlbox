
import sys
import h5py
import json
import tensorflow as tf
from drlbox.layer.noisy_dense import NoisyDenseIG, NoisyDenseFG


TASKER_KWARGS = dict(
    env_maker=None,
    state_to_input=None,
    load_model=None,
    load_model_custom=None,
    verbose=False,
    )

class Tasker:

    KWARGS = TASKER_KWARGS

    def __init__(self, **kwargs):
        # combine arguments from self.KWARGS and kwargs and set arguments
        self.__dict__.update(self.KWARGS)
        self.__dict__.update(kwargs)
        self.print_kwargs(self.__dict__, 'All arguments', default=self.KWARGS)

        # set self.state_to_input to 'do nothing' if it is None
        if self.state_to_input is None:
            self.state_to_input = lambda x: x

    def print_kwargs(self, kwargs, header=None, default=None):
        self.print('#### {} ####'.format(header))
        for key, value in sorted(kwargs.items()):
            statement = '    {} = {}'.format(key, value)
            if default is not None:
                if key not in default:
                    statement += ' (UNUSED)'
                    print('Warning: {} = {} set but unused'.format(key, value),
                          file=sys.stderr, flush=True)
            self.print(statement)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)

    def do_load_model(self, load_weights=True):
        custom_objects = {'NoisyDenseIG': NoisyDenseIG,
                          'NoisyDenseFG': NoisyDenseFG}
        if self.load_model_custom is not None:
            custom_objects.update(self.load_model_custom)
        if load_weights:
            return tf.keras.models.load_model(self.load_model, custom_objects)
        else:
            return self.load_model_no_weights(self.load_model, custom_objects)

    def load_model_no_weights(self, filepath, custom_objects=None):
        if custom_objects is None:
            custom_objects = {}
        with h5py.File(filepath, mode='r') as f:
            # instantiate model
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError('No model found in config file.')
            model_config = json.loads(model_config.decode('utf-8'))
            model = tf.keras.models.model_from_config(model_config,
                custom_objects=custom_objects)
        return model

