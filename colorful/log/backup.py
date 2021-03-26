import torch
from colorful.log.listener import Listener
import datetime
import os
import json

class Backup(Listener):
    def __init__(self, save_every, save_dir, config, config_file='config.json', save_file='backup_model.pth'):
        super(Backup, self).__init__(save_every)
        self.save_dir = save_dir
        self.config_file = config_file
        self.save_file = save_file
        self.config = config

        if not os.path.isdir(save_dir):
            raise RuntimeError(f"Directory '{save_dir}' does not exist!")

    def log(self, iter, loss, network):
        path = os.path.join(self.save_dir, self.save_file)
        torch.save(network.state_dict(), path)
        with open(os.path.join(self.save_dir, self.config_file), 'w') as file:
            config = {"time":str(datetime.datetime.now()), "iter": iter, "config": self.config}
            file.write(json.dumps(config, indent=2))

