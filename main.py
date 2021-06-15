import argparse
import json

import torch.cuda

from util import util

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Dispatcher for training colorization models")
    parser.add_argument('config', metavar='config', type=str, help="Config file path")

    # Get args and extras
    args, extras = parser.parse_known_args()
    # Key value pairs to override config data
    # Eg. to override the learning rate of the solver add the 2 arguments to the call
    #   base.solver.config.optimizer.args.lr 1e-5
    # Full:
    #   python3.7 main.py train colorful.json base.solver.config.optimizer.args.lr 1e-5
    override = {key: (key.split("."), value) for key, value in zip(extras[::2], extras[1::2])}

    with open(args.config, 'r') as file:
        config = json.load(file)

    # Overriding of config
    try:
        for key in override:
            path, val = override[key]
            final_key = path[-1]

            tmp = config
            for p in path[:-1]:
                tmp = tmp[p]
            val_type = type(tmp[final_key])
            tmp[final_key] = val_type(val)
    except ValueError:
        # noinspection PyUnboundLocalVariable
        print(f"Value error overriding default config for key: {key}")
        exit()

    # Create model
    model = util.factory(config['model'])

    # Create solver
    solver_data = config['solver']
    solver_class = util.import_attr(solver_data['class'])
    solver = solver_class(model, config)

    solver.train()

