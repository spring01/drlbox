
import os
import sys
import argparse
import importlib
import subprocess


DEF_ENV = 'drlbox/env/default.py'
DEF_FEATURE = 'drlbox/feature/fc.py'

'''
Manager class of argparse, config, env, model
'''
class Manager:

    def __init__(self, description, default_config):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--load_weights', default=None,
            help='If specified, load weights and start training from there')
        parser.add_argument('--save', default='./output',
            help='Directory to save data to')

        # user-definable imports
        parser.add_argument('--import_path', nargs='+', default=[os.getcwd()],
            help='path where the user-defined scripts are located')
        parser.add_argument('--env', nargs='+',
            default=['CartPole-v0'],
            help='openai gym environment.')
        parser.add_argument('--feature', nargs='+',
            default=['200 100'],
            help='neural network feature builder')
        parser.add_argument('--config', default=default_config,
            help='algorithm configurations')

        self.default_config = default_config
        self.parser = parser

    def import_files(self):
        # parse arguments
        args = self.parser.parse_args()
        print('########## All arguments:', args)
        self.args = args

        # dynamically import net and interface
        for path in args.import_path:
            sys.path.append(path)
        config_def = importlib.import_module(parse_import(self.default_config))
        config = importlib.import_module(parse_import(args.config))

        # set default configurations in config
        keys_config = dir(config)
        for key_def, value_def in vars(config_def).items():
            if key_def not in keys_config:
                setattr(config, key_def, value_def)
        self.config = config

        # import and setup env
        if len(args.env) > 1:
            env_import, env_args = args.env[-1], args.env[:-1]
        else:
            env_import, env_args = DEF_ENV, args.env
        env_spec = importlib.import_module(parse_import(env_import))
        self.env, self.env_name = env_spec.make_env(*env_args)

        # import and setup feature network
        if len(args.feature) > 1:
            feature_import, feature_args = args.feature[-1], args.feature[:-1]
        else:
            feature_import, feature_args = DEF_FEATURE, args.feature
        self.feature_args = feature_args
        feature_spec = importlib.import_module(parse_import(feature_import))
        self.feature_builder = feature_spec.feature
        self.state_to_input = feature_spec.state_to_input

    def build_model(self, model_builder):
        feature_args = self.env.observation_space, *self.feature_args
        state, feature = self.feature_builder(*feature_args)
        return model_builder(state, feature, self.env.action_space)

    def get_output_folder(self):
        parent_dir = self.args.save
        experiment_id = 0
        if not os.path.isdir(parent_dir):
            subprocess.call(['mkdir', '-p', parent_dir])
            print('Made output dir', parent_dir)
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('-run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, self.env_name)
        parent_dir += '-run{}'.format(experiment_id)
        subprocess.call(['mkdir', '-p', parent_dir])
        return parent_dir


def parse_import(filename):
    import_name, _ = os.path.splitext(filename)
    return import_name.replace('/', '.')


