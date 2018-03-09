
import h5py
import json
import tensorflow as tf
from drlbox.layer.noisy_dense import NoisyDenseIG


class Tasker:

    KEYWORD_DICT = dict(env_maker=None,
                        state_to_input=None,
                        load_model=None,
                        load_model_custom=None,
                        noisynet=None,
                        verbose=False,
                        )

    def __init__(self, **kwargs):
        # combine arguments from self.KEYWORD_DICT and kwargs and set arguments
        self.__dict__.update(self.KEYWORD_DICT)
        self.__dict__.update(kwargs)

        # print arguments
        self.print('#### All arguments ####')
        for keyword, value in sorted(self.__dict__.items()):
            statement = '    {} = {}'.format(keyword, value)
            if keyword not in self.KEYWORD_DICT:
                statement += ' (UNUSED)'
            self.print(statement)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)

    def do_load_model(self, load_weights=True):
        custom_objects = {}
        if self.noisynet is not None:
            noisy_layer_dict = {'NoisyDenseIG': NoisyDenseIG}
            custom_objects.update(noisy_layer_dict)
        if self.load_model_custom is not None:
            custom_objects.update(self.load_model_custom)
        if load_weights:
            return tf.keras.models.load_model(self.load_model, custom_objects)
        else:
            return self.load_model_no_weights(self.load_model, custom_objects)

    def load_model_no_weights(self, filepath, custom_objects=None):
        if not custom_objects:
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

