import os
import re
import time
import traceback

import numpy as np
import pandas as pd

from com.designingnn.cnn.ModelRunner import ModelRunner
from com.designingnn.core import AppContext
from com.designingnn.resources import mnist_state_space_parameters, mnist_hyper_parameters
from com.designingnn.rl.QLearner import QLearner
from com.designingnn.rl.QValues import QValues


class DesignNeuralNetwork:
    def __init__(self):
        self.replay_columns = ['net',  # Net String
                               'accuracy_best_val',
                               'iter_best_val',
                               'accuracy_last_val',
                               'iter_last_val',
                               'accuracy_best_test',
                               'accuracy_last_test',
                               'ix_q_value_update',  # Iteration for q value update
                               'epsilon',  # For epsilon greedy
                               'time_finished',  # UNIX time
                               'machine_run_on']

        if AppContext.DATASET == 'mnist':
            self.state_space_parameters = mnist_state_space_parameters
            self.hyper_parameters = mnist_hyper_parameters
        elif AppContext.DATASET == '':
            pass

        self.list_path = os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'data'), AppContext.DATASET)
        self.replay_dictionary_path = os.path.join(self.list_path, 'replay_database.csv')
        self.replay_dictionary, self.q_training_step = self.load_replay(self.replay_dictionary_path)

        self.epsilon = self.state_space_parameters.epsilon_schedule[0][0]
        self.number_models = self.state_space_parameters.epsilon_schedule[0][1]

        self.number_q_updates_per_train = 100

        self.schedule_or_single = True

        self.qlearner = self.load_qlearner()
        self.check_reached_limit()

    def start_app(self):
        while not self.check_reached_limit():
            net_to_run, iteration = self.generate_new_netork()

            print 'Ready to train ' + net_to_run

            checkpoint_dir = os.path.join(self.list_path, 'checkpoints')

            model_dir = self.get_model_dir(checkpoint_dir, net_to_run)

            trainer = ModelRunner(model_dir, self.hyper_parameters, self.state_space_parameters)

            train_out = trainer.run_one_model(net_to_run, iteration)
            print 'OUT', train_out

            (iter_best, acc_best) = max(train_out['test_accs'].items(), key=lambda x: x[1])
            (iter_last, acc_last) = max(train_out['test_accs'].items(), key=lambda x: x[0])

            # Clear out model files
            self.clear_logs(checkpoint_dir,
                            pd.DataFrame({'net': [net_to_run],
                                          'iter_best_val': [iter_best],
                                          'iter_last_val': [iter_last]}))

            self.incorporate_trained_net(net_to_run,
                                         acc_best,
                                         iter_best,
                                         acc_last,
                                         iter_last,
                                         self.epsilon,
                                         [self.q_training_step],
                                         'localhost')
        print 'EXPERIMENT COMPLETE!'

    def clear_logs(self, ckpt_dir, replay):
        ''' Deletes uneeded log files and model saves:
            args:
                replay - with standard replay dic columns. Deletes only model saves that aren't for the first iteration,
                            the best iteration, and the last iteration
                ckpt_dir - where the models are saved, it must have a filemap
        '''
        file_map = pd.read_csv(os.path.join(ckpt_dir, 'file_map.csv'))

        for i in range(len(file_map)):
            net = file_map.net.values[i]

            # First check that the model was run on this computer
            if net not in replay.net.values:
                continue
            else:
                folder_path = os.path.join(ckpt_dir, str(int(file_map[file_map.net == net].file_number.values[0])))
                if not os.path.isdir(folder_path):
                    continue
                best_iter = replay[replay.net == net].iter_best_val.values[0]
                last_iter = replay[replay.net == net].iter_last_val.values[0]
                model_saves = [f for f in os.listdir(folder_path) if f.find('modelsave_iter') >= 0]

                # make sure that the model actual ran
                if not model_saves:
                    continue
                first_iter = min([int(re.split('_|\.', f)[2]) for f in model_saves])
                model_saves_to_keep = ['modelsave_iter_%i.solverstate' % iteration for iteration in
                                       [best_iter, last_iter, first_iter]] + \
                                      ['modelsave_iter_%i.caffemodel' % iteration for iteration in
                                       [best_iter, last_iter, first_iter]]

                # make sure all of the files are there
                if not np.all(
                        [os.path.isfile(os.path.join(folder_path, savefile)) for savefile in model_saves_to_keep]):
                    continue

                # Delete extraneous files
                for f in model_saves:
                    if f not in model_saves_to_keep:
                        os.remove(os.path.join(folder_path, f))

    def get_model_dir(self, base_ckpt_dir, net):
        if not os.path.exists(base_ckpt_dir):
            os.makedirs(base_ckpt_dir)

        ckpt_file_map_file = os.path.join(base_ckpt_dir, 'file_map.csv')
        if not os.path.exists(ckpt_file_map_file):
            pd.DataFrame(columns=['net', 'file_number']).to_csv(ckpt_file_map_file, index=False)

        ckpt_file_map = pd.read_csv(ckpt_file_map_file)

        # check if we already have a folder
        if sum(ckpt_file_map['net'] == net) == 0:
            next_ckpt = 1 if len(ckpt_file_map) == 0 else max(ckpt_file_map['file_number']) + 1
            ckpt_file_map = pd.concat([ckpt_file_map, pd.DataFrame({'net': [net], 'file_number': [next_ckpt]})])
            ckpt_file_map.to_csv(ckpt_file_map_file, index=False)
        else:
            next_ckpt = ckpt_file_map[ckpt_file_map['net'] == net]['file_number'].values[0]

        return os.path.join(base_ckpt_dir, str(int(next_ckpt)))

    def load_replay(self, replay_dictionary_path):
        if os.path.isfile(replay_dictionary_path):
            print 'Found replay dictionary'
            replay_dic = pd.read_csv(replay_dictionary_path)
            q_training_step = max(replay_dic.ix_q_value_update)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic, q_training_step

    def load_qlearner(self):
        # Load previous q_values
        if os.path.isfile(os.path.join(self.list_path, 'q_values.csv')):
            print 'Found q values'
            qstore = QValues()
            qstore.load_q_values(os.path.join(self.list_path, 'q_values.csv'))
        else:
            qstore = None

        ql = QLearner(self.state_space_parameters,
                      self.epsilon,
                      qstore=qstore,
                      replay_dictionary=self.replay_dictionary)

        return ql

    def filter_replay_for_first_run(self, replay):
        ''' Order replay by iteration, then remove duplicate nets keeping the first'''
        temp = replay.sort_values(['ix_q_value_update']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['net'])

    def number_trained_unique(self, epsilon=None):
        '''Epsilon defaults to the minimum'''
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)

    def check_reached_limit(self):
        ''' Returns True if the experiment is complete
        '''
        if len(self.replay_dictionary):
            completed_current = self.number_trained_unique(self.epsilon) >= self.number_models

            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_trained_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False

                            break
                else:
                    completed_experiment = True

                return completed_experiment
            else:
                return False

    def generate_new_netork(self):
        try:
            (net,
             acc_best_val,
             iter_best_val,
             acc_last_val,
             iter_last_val,
             acc_best_test,
             acc_last_test,
             machine_run_on) = self.qlearner.generate_net()

            # We have already trained this net
            if net in self.replay_dictionary.net.values:
                self.q_training_step += 1
                self.incorporate_trained_net(net,
                                             acc_best_val,
                                             iter_best_val,
                                             acc_last_val,
                                             iter_last_val,
                                             self.epsilon,
                                             [self.q_training_step],
                                             machine_run_on)
                return self.generate_new_netork()
            else:
                self.q_training_step += 1
                return net, self.q_training_step

        except Exception:
            print traceback.print_exc()

    def incorporate_trained_net(self,
                                net_string,
                                acc_best_val,
                                iter_best_val,
                                acc_last_val,
                                iter_last_val,
                                epsilon,
                                iters,
                                machine_run_on):

        try:
            # If we sampled the same net many times, we should add them each into the replay database
            for train_iter in iters:
                replay_dict = {
                    'net': [net_string],
                    'accuracy_best_val': [acc_best_val],
                    'iter_best_val': [iter_best_val],
                    'accuracy_last_val': [acc_last_val],
                    'iter_last_val': [iter_last_val],
                    'accuracy_best_test': [-1.0],
                    'accuracy_last_test': [-1.0],
                    'ix_q_value_update': [train_iter],
                    'epsilon': [epsilon],
                    'time_finished': [time.time()],
                    'machine_run_on': [machine_run_on]
                }

                self.replay_dictionary = pd.concat([self.replay_dictionary, pd.DataFrame(replay_dict)])
                self.replay_dictionary.to_csv(self.replay_dictionary_path, index=False, columns=self.replay_columns)

            self.qlearner.update_replay_database(self.replay_dictionary)

            for i in range(len(iters)):
                self.qlearner.sample_replay_for_update()

            self.qlearner.save_q(self.list_path)

        except Exception:
            print traceback.print_exc()


if __name__ == '__main__':
    AppContext.TRAINED_MODELS_PATH = '/mnt/D/Learning/MTSS/Sem 4/code/designing-neural-networks/trained_models/mnist'
    AppContext.DATASET_PATH = '/mnt/D/Learning/MTSS/Sem 4/code/designing-neural-networks/data/mnist'

    DesignNeuralNetwork().start_app()
