import pandas as pd

from com.designingnn.server.service.rl.State import State


class QValues:
    ''' Stores Q_values with helper functions.'''

    def __init__(self):
        self.q = {}

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_layer_type',
                                                                'start_layer_depth',
                                                                'start_filter_depth',
                                                                'start_filter_size',
                                                                'start_stride',
                                                                'start_image_size',
                                                                'start_fc_size',
                                                                'start_terminate',
                                                                'end_layer_type',
                                                                'end_layer_depth',
                                                                'end_filter_depth',
                                                                'end_filter_size',
                                                                'end_stride',
                                                                'end_image_size',
                                                                'end_fc_size',
                                                                'end_terminate',
                                                                'utility']]):
            start_state = State(layer_type=row[0],
                                layer_depth=row[1],
                                filter_depth=row[2],
                                filter_size=row[3],
                                stride=row[4],
                                image_size=row[5],
                                fc_size=row[6],
                                terminate=row[7]).as_tuple()
            end_state = State(layer_type=row[8],
                              layer_depth=row[9],
                              filter_depth=row[10],
                              filter_size=row[11],
                              stride=row[12],
                              image_size=row[13],
                              fc_size=row[14],
                              terminate=row[15]).as_tuple()
            utility = row[16]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)

    def save_to_csv(self, q_csv_path):
        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_image_size = []
        start_fc_size = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_image_size = []
        end_fc_size = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = State(state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_image_size.append(start_state.image_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_image_size.append(to_state.image_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame({'start_layer_type': start_layer_type,
                              'start_layer_depth': start_layer_depth,
                              'start_filter_depth': start_filter_depth,
                              'start_filter_size': start_filter_size,
                              'start_stride': start_stride,
                              'start_image_size': start_image_size,
                              'start_fc_size': start_fc_size,
                              'start_terminate': start_terminate,
                              'end_layer_type': end_layer_type,
                              'end_layer_depth': end_layer_depth,
                              'end_filter_depth': end_filter_depth,
                              'end_filter_size': end_filter_size,
                              'end_stride': end_stride,
                              'end_image_size': end_image_size,
                              'end_fc_size': end_fc_size,
                              'end_terminate': end_terminate,
                              'utility': utility})
        q_csv.to_csv(q_csv_path, index=False)
