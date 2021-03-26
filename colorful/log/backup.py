import torch
from colorful.log.listener import Listener
import datetime
import os
import json

class Backup(Listener):
    def __init__(self, save_every, save_dir, config, state_file='state.json', config_file='config.pth', save_file='backup_model.pth'):
        super(Backup, self).__init__(save_every)
        self.save_dir = save_dir
        self.config_file = config_file
        self.state_file = state_file
        self.save_file = save_file
        self.config = config

        if not os.path.isdir(save_dir):
            raise RuntimeError(f"Directory '{save_dir}' does not exist!")

    def log(self, iter, loss, network):
        path = os.path.join(self.save_dir, self.save_file)
        torch.save(network.state_dict(), path)
        with open(os.path.join(self.save_dir, self.state_file), 'w') as file:
            data = {"time":str(datetime.datetime.now()), "iter": iter}
            file.write(json.dumps(data, indent=2))
        torch.save(self.config, os.path.join(self.save_dir, self.config_file))

