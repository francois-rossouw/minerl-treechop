import os
import csv
import pandas as pd
import numpy as np
import json
from datetime import datetime
import copy
import torch

from utils.gen_args import Arguments


class DataWriter:
    """
    Object for logging results of training / evaluation
    """
    def __init__(self, args: Arguments):
        self.start_time = datetime.now()
        self.folder = 'runs_recordings'
        self._set_sub_folder()
        if not os.path.exists(self.folder):  # Create folder if needed
            os.makedirs(self.folder)
        self.csv_file = self._get_filename('run_stats', 'csv')
        self.json_file = self._get_filename('args', 'json')
        self.pretrain_csv = self._get_filename('pretrain', 'csv')
        self.column_names = ['Episode', 'Timestamp', 'Reward', 'Expert observations', 'Loss', '30 step mean reward',
                             'std', 'min', '25%', '50%', '75%', 'max', 'Episode steps', 'Total steps']
        self.pretrain_columns = ['Step', 'Loss', 'Btn Acc', 'Cam Acc']
        self.total_steps = 0
        self._write_args(args)

        self._write_data(self.csv_file, data=self.column_names, cmd='w')  # Create new file and write
        self._write_data(self.pretrain_csv, data=self.pretrain_columns, cmd='w')  # Create new file and write

    def _write_data(self, filename, data, cmd='a+'):
        with open(filename, cmd, newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def write_episode_data(self, ep, timestamp, reward_hist, exp_percent, loss, total_steps):
        df = pd.DataFrame(list(reward_hist))
        # Extract reward history statistics
        cnt, mean, std, min_v, low_quartile, median, upper_quartile, max_v = df.describe().iloc[:][0]
        data = [ep, timestamp, reward_hist[-1], exp_percent, loss, mean, std, min_v, low_quartile,
                median, upper_quartile, max_v, total_steps-self.total_steps, total_steps]
        self.total_steps = total_steps
        self._write_data(self.csv_file, data)

    def write_pretrain_data(self, step, loss, btn_acc, cam_acc):
        data = [step, loss, btn_acc, cam_acc]
        self._write_data(self.pretrain_csv, data)

    def _set_sub_folder(self):
        sub_folder = self.start_time.strftime("%d-%m_%H:%M")
        self.folder = '/'.join([self.folder, sub_folder])

    def _get_filename(self, filename, ext):
        filename = '/'.join([self.folder, filename])
        filename = '.'.join([filename, ext])
        return filename

    def _write_args(self, args):
        all_args = copy.deepcopy(args.as_dict())
        # for key, value in all_args.items():
        #     if isinstance(value, np.ndarray)
        with open(self.json_file, 'w') as outfile:
            json.dump(all_args, outfile, default=default, indent=2)
        # args.save('args.json')


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    elif type(obj).__module__ == torch.__name__:
        if isinstance(obj, torch.device):
            return ':'.join([obj.type, str(obj.index)])
        else:
            return obj
    raise TypeError('Unknown type:', type(obj))


if __name__ == '__main__':
    args = Arguments(underscores_to_dashes=True).parse_args(known_only=True)
    dw = DataWriter(args=args)
    dw.write_episode_data(
        ep=1,
        timestamp='01h 54m 12s',
        reward_hist=list([64, 32, 48, 29, 16, 51, 61, 49, 59, 10, 1, 65]),
        exp_percent='100.00 %',
        loss=0.1629,
        total_steps=122
    )
    dw.write_episode_data(
        ep=2,
        timestamp='01h 55m 11s',
        reward_hist=list([32, 48, 29, 16, 51, 61, 49, 59, 10, 1, 65, 32]),
        exp_percent='97.23 %',
        loss=0.1623,
        total_steps=2122
    )
