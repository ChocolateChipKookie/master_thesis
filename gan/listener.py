from torch.utils.data import DataLoader

from util.listener import SolverListener
import sys
import datetime
import torch
import os
import copy
import json


class OutputLogger(SolverListener):
    def __init__(self, solver, frequency=1, output=None):
        """
        Logs basic output for monitoring the solver status
        :param solver: The solver that is being monitored
        :param frequency: The frequency of logging
        :param output: Output file path, if none stdout is used as default
        """
        super(OutputLogger, self).__init__(solver, frequency)

        if output:
            self.output = open(output, 'a' if self.solver.restore else 'w')
        else:
            self.output = sys.stdout

        self.iterations = self.solver.iterations
        self.begin = datetime.datetime.now()
        self.iterations_magnitude = len(str(self.iterations))

    def update(self, iter, loss):
        # Get current time
        now = datetime.datetime.now()
        # Create string
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        # Fetch hours, minutes and seconds since begin
        since_begin = now - self.begin
        total_seconds = since_begin.seconds
        days = since_begin.days
        hours = total_seconds // 3600 + days * 24
        minutes = (total_seconds - hours * 3600) // 60
        seconds = total_seconds - hours * 3600 - minutes * 60
        # Format
        since_begin_str = f"{hours:03}:{minutes:02}:{seconds:02}"
        # Justify, output and flush
        iter_str = str(iter).rjust(self.iterations_magnitude)

        loss_D, loss_D_real, loss_D_fake, loss_G, loss_G_cond, loss_G_fake = loss

        self.output.write(
            f"[{now_str}] [{since_begin_str}] [{iter_str}/{self.iterations}] [ D: {loss_D:>12.4e} | real: {loss_D_real:>12.4e} | fake: {loss_D_fake:>12.4e}] [G: {loss_G:>12.4e} | cond: {loss_G_cond:>12.4e} | fake: {loss_G_fake:>12.4e}]\n"
        )
        self.output.flush()


class LossLogger(SolverListener):
    def __init__(self, solver, frequency=1, format="{0}.\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n", output = None):
        """
        Logs loss
        :param solver: Solver to be monitored
        :param frequency: Frequence of logging
        :param format: Loss format, {0} for iteration and {1} for loss
        :param output: Output file path, stdout if none is defined
        """
        super(LossLogger, self).__init__(solver, frequency)
        if output:
            if self.solver.restore:
                self.output = open(output, 'a')
            else:
                self.output = open(output, 'w')
        else:
            self.output = sys.stdout

        self.format = format

    def update(self, iter, loss):
        # Write and flush
        self.output.write(self.format.format(iter, *loss))
        self.output.flush()


class Backup(SolverListener):
    def __init__(self, solver, frequency, save_dir, config_file='backup_config.json', generator_file='backup_generator.pth', discriminator_file='backup_discriminator.pth'):
        """
        Backs up the network in regular periods defined by the frequency
        :param solver: Solver to be monitored
        :param frequency: Frequency of backups
        :param save_dir: Base directory for backup files
        :param config_file: Config file name
        :param save_file: Model backup file
        """
        super(Backup, self).__init__(solver, frequency)
        self.save_dir = save_dir
        self.config_file = config_file
        self.generator_file = generator_file
        self.discriminator_file = discriminator_file

        if not os.path.isdir(save_dir):
            raise RuntimeError(f"Directory '{save_dir}' does not exist!")

    def update(self, iter, loss):
        # Save state dict of model
        path_g = os.path.join(self.save_dir, self.generator_file)
        torch.save(self.solver.net_G.state_dict(), path_g)
        path_d = os.path.join(self.save_dir, self.discriminator_file)
        torch.save(self.solver.net_D.state_dict(), path_d)

        with open(os.path.join(self.save_dir, self.config_file), 'w') as file:
            # Deepcopy the config, so it does not get modified
            config = copy.deepcopy(self.solver.config)
            # Create state
            state = {
                "time": str(datetime.datetime.now()),
                "iter": iter,
                "state_dict": path_g,
                "discriminator_state_dict": path_d,
            }
            # Update config
            config['state'] = state
            config['restore'] = True
            # Dump JSON
            json.dump(config, file, indent=2)


class Validator(SolverListener):
    def __init__(self, solver, frequency, output=None, save_every=None, snapshot_dir=None):
        super(Validator, self).__init__(solver, frequency)
        self.save = save_every is not None

        if self.save:
            if not snapshot_dir:
                raise RuntimeError("Destination directory not specified")
            self.save_every = save_every
            self.snapshot_dir = snapshot_dir

            if not os.path.exists(snapshot_dir):
                raise RuntimeError("Snapshot directory does not exist")
            if not os.path.isdir(snapshot_dir):
                raise RuntimeError("Defined snapshot directory path is not a directory")

        if output:
            self.output = open(output, 'a' if self.solver.restore else 'w')
        else:
            self.output = sys.stdout
        self.counter = 0

    def write(self, str):
        self.output.write(str)
        self.output.flush()

    def update(self, iter, loss):
        # If it is the first iteration, skip
        if iter == 0:
            return
        # Note the beginning of the validation
        begin = datetime.datetime.now()
        print("===============================================")
        print(f"Validating...")

        # For simpler code
        solver = self.solver

        # Fetch sampler and dataloader
        sampler = solver.val_sampler
        data_loader = DataLoader(solver.val_dataset, batch_size=solver.batch_size, num_workers=solver.loaders, sampler=sampler)

        # Define initial variables
        total_validated = 0
        total_i = 0.
        total_losses = [0, 0, 0, 0, 0, 0]
        # Validate all
        for batch, _ in data_loader:
            total_i += 1
            total_validated += batch.shape[0]

            with torch.no_grad():
                # Fetch input for the network
                l = batch[:, :1, :, :].to(solver.device)
                l_norm = solver.net_G.normalize_l(l)
                ab = batch[:, 1:, :, :].to(solver.device)
                ab_norm = solver.net_G.normalize_ab(ab)

                # Create fake batch
                fake = solver.net_G(l_norm, True)
                solver.eval_D(l_norm, ab_norm, fake)
                solver.eval_G(l_norm, ab_norm, fake)

                losses = (x.item() for x in self.solver.calculate_loss(batch))
                for i, loss in enumerate(losses):
                    total_losses[i] += loss
                # Clean memory
                torch.cuda.empty_cache()

        # Calculate results
        avg_losses = []
        for loss in total_losses:
            avg_losses.append(loss/total_i)

        end = datetime.datetime.now()
        total_duration = end - begin
        total_seconds = total_duration.total_seconds()

        print(f"Validated {total_validated} samples!")
        print(f"Time spent: {int(total_seconds)} sec")
        print(f"Iteration: {iter}")
        print(f"Average loss: [ D {avg_losses[0]:>12.4e} | real: {avg_losses[1]:>12.4e} | fake: {avg_losses[2]:>12.4e}] [ G: {avg_losses[3]:>12.4e} | cond: {avg_losses[4]:>12.4e} | fake: {avg_losses[5]:>12.4e}]")

        self.counter += 1
        if self.save and self.counter % self.save_every == 0:
            now = datetime.datetime.now()
            now_str = now.strftime("%m_%d")
            name_G = f"G-{now_str}-{iter}.pth"
            name_D = f"D-{now_str}-{iter}.pth"
            torch.save(self.solver.net_G.state_dict(), os.path.join(self.snapshot_dir, name_G))
            torch.save(self.solver.net_D.state_dict(), os.path.join(self.snapshot_dir, name_D))
            print(f"Saved snapshot {name_G} to {self.snapshot_dir}")
        print("===============================================")

        self.write(f"{iter:<8} {avg_losses[0]} {avg_losses[1]} {avg_losses[2]} {avg_losses[3]} {avg_losses[4]} {avg_losses[5]}\n")
