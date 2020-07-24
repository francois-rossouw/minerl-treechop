"""
Checkpointing functions to save model every x epochs.
"""

import os
import shutil
import torch

FOLDER = 'train/'


def load_model(args, arch, model, optimizer):
    """Loads model from storage.
    Arguments:
        args {argparse.Namespace} -- Input arguments for storage of constants
        model {[nets.PolicyLinear, nets.PolicyCNN]} -- Model to replace state_dict into
        optimizer {torch.optim.adam.Adam} -- Optimizer to load state_dict into
    """

    filename = ''.join([FOLDER, arch, '.pt'])
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer' in checkpoint:
            if not isinstance(optimizer, tuple):
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                state_dicts = checkpoint['optimizer']
                for optim, state_dict in zip(optimizer, state_dicts):
                    optim.load_state_dict(state_dict)
        print(f"=> loaded {arch} checkpoint")
    else:
        print(f"\tNo checkpoint found in folder '{FOLDER}', doing cold start.")


def get_av_score(args):
    """Loads model from storage.
    Arguments:
        args {argparse.Namespace} -- Input arguments for storage of constants
        model {[nets.PolicyLinear, nets.PolicyCNN]} -- Model to replace state_dict into
        optimizer {torch.optim.adam.Adam} -- Optimizer to load state_dict into
    """

    filename = ''.join([FOLDER, 'model_best.pth.tar'])
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=args.device)
        return checkpoint['other'].av_score
    else:
        print(f"\tNo checkpoint found in folder '{FOLDER}', doing cold start.")
        return None


def save_checkpoint(arch, state, is_best):
    """Saves state of models at certain checkpoints.
    Arguments:
        arch {str} -- String describing model architecture (fc / cnn)
        state {dict} -- Dictionary containing current epoch, architecture type,
                        model- and optimizer state
        is_best {bool} -- Is this the the best model trained
    """

    filename = ''.join([FOLDER, arch, '.pt'])
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, ''.join([FOLDER, 'model_best.pth.tar']))
