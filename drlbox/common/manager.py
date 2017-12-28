
import os
import sys
import argparse
import importlib
import subprocess


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
        parser.add_argument('--import_env', nargs='+',
            default=['drlbox.env.default', 'CartPole-v0'],
            help='openai gym environment.')
        parser.add_argument('--import_feature', nargs='+',
            default=['drlbox.feature.fc', '200 100'],
            help='neural network feature builder')
        parser.add_argument('--import_config', default=default_config,
            help='algorithm configurations')

        self.default_config = default_config
        self.parser = parser

    def parse_import(self):
        # parse arguments
        args = self.parser.parse_args()
        print('########## All arguments:', args)
        self.args = args

        # dynamically import net and interface
        for path in args.import_path:
            sys.path.append(path)
        config_def = importlib.import_module(self.default_config)
        config = importlib.import_module(args.import_config)

        # set default configurations in config
        for key, value in config_def.__dict__.items():
            if key not in config.__dict__:
                config.__dict__[key] = value
        self.config = config

        env_spec = importlib.import_module(args.import_env[0])
        self.env, self.env_name = env_spec.make_env(*args.import_env[1:])

        feature_spec = importlib.import_module(args.import_feature[0])
        self.feature_builder = feature_spec.feature
        self.state_to_input = feature_spec.state_to_input

    def build_model(self, model_builder):
        feature_args = self.env.observation_space, *self.args.import_feature[1:]
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

