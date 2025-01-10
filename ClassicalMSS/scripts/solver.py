# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs
"""Main training loop for the project"""

import logging
from pathlib import Path
import sys

from dora import get_xp # The facebook tool for experiment management
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
from typing import Dict
import torch
import torch.nn.functional as F

root = Path(__file__).parent.parent
sys.path.append(str(root))
from ClassicalMSS.utils import states, pretrained # No audio augmentation or distributed training for now
from ClassicalMSS.utils.ema import ModelEMA
from ClassicalMSS.utils.svd import svd_penalty
from ClassicalMSS.utils.evaluate import evaluate
from ClassicalMSS.utils.utils import pull_metric

logger = logging.getLogger(__name__)

def _summary(metrics):
    # function name starts with underscore to indicate it is a private function
    # although it is not strictly private, but it can not be imported by "import *"
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())

class Solver:
    def __init__(self, loaders, model, optimizer, args):
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        # May set up quantizer later if the dictionary is too large
        # May set up distributed training later when the model is too large for one A100 GPU
        self.device = next(iter(model.parameters())).device

        # --------------------------------------------------------------------------
        """EMA Setup"""
        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self.emas = {'batch': [], 'epoch': []}
        for ema_type in self.emas.keys():
            decays = getattr(args.ema, ema_type)
            # Updating the EMA weight once every epoch is scarce, so we can afford the cost of 
            # switching from GPU to CPU. Putting the EMA on CPU can save memory on GPU.
            device = self.device if ema_type == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[ema_type].append(ModelEMA(self.model, decay, device=device))
        
        # --------------------------------------------------------------------------
        """Data Augmentation Setup"""
        # Data augmentation is temporarily not used

        # --------------------------------------------------------------------------
        """Experiment Tracking Setup"""
        xp = get_xp() # xp is an experiment tracking object, the specific experiment get_xp() gets is globally set outside of this class
                        # It is recommended to set the experiment unique for each train, only use the same experiment for different runs of the same train
        self.folder = xp.folder # The folder where the experiment (model states and checkpoints) is stored
        self.checkpoint_file = xp.folder / 'checkpoint.th' # "th" stands for "torch", similar to "pth" (pytorch)
        self.best_file = xp.folder / 'best.th'
        # The resolve() method is used to get the actual path of the file (not something like a symbolic link)
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False
        # xp.link helps to track and store experiment metrics and history
        self.link = xp.link
        self.history = xp.history

        # --------------------------------------------------------------------------
        self._reset()
    
    def _serialize(self, epoch):
        """
        This method is responsible for saving (or serializing) the model state and optimizer state:
        The current model state is saved under checkpoint.th (which will be overwritten every epoch, just for resuming training)
        Every certain number of epochs, the current model state is saved under checkpoint_{epoch+1}.th
        These two saves contains EMA models, but not the best model
        The best model state is saved under best.th (which will be overwritten every time a new best model is found)
        """
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for ema_type, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{ema_type}_{k}'] = ema.state_dict()
        
        # Safely save the checkpoint
        # The write_and_rename function from dora.utils is used to save the checkpoint file as a tmp file and rename it to the checkpoint file
        # Since the rename process is atomic but the writing (torch.save) is time-consuming, we use this function to avoid corrupted checkpoints
        # Source code of which is:
        ## @contextmanager
        ## def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp"):
        ## tmp_path = str(path) + suffix
        ## with open(tmp_path, mode) as f:
        ## yield f
        ## os.rename(tmp_path, path)
        with write_and_rename(self.checkpoint_file, self.folder) as tmp:
            torch.save(package, tmp)
        
        save_every = self.args.save_every
        if save_every and (epoch+1) % save_every == 0 and epoch+1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch+1}.th') as tmp:
                torch.save(package, tmp)
        
        if self.best_changed:
            # Saving only the latest best model
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state # The serialize_model function saves the current state, here change it to the best state
                torch.save(package, tmp)
            self.best_changed = False


    def _reset(self):
        """
        Reset state of the solver, potentially using checkpoints of the current solver, pre-trained models,
        or continue from a previous saved checkpoint state
        """
        if self.checkpoint_file.exists(): # Check if the checkpoint file exists (the directory indeed leads to a file)
            logger.info(f"Loading checkpoint model: {self.checkpoint_file}")
            package = torch.load(self.checkpoint_file, 'cpu')
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history'] # Replace all elements in the history list with the ones in the checkpoint
            self.best_state = package['best_state']
            for ema_type, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{ema_type}_{k}'])
        elif self.args.continue_pretrained:
            model = pretrained.get_model(
                name=self.args.continue_pretrained,
                repo=self.args.pretrained_repo)
            self.model.load_state_dict(model.state_dict())
        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name # The path to the checkpoint file
            logger.info("Loading checkpoint model from: %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_from_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_optimizer:
                self.optimizer.load_state_dict(package['optimizer'])
    
    def _format_train(self, metrics: Dict) -> Dict:
        """Formatting for train/valid metrics."""
        losses = { # The two essential metrics for training
            'loss': format(metrics['loss'], '.4f'), # The format() function converts a value into a formatted string representation
            'reco': format(metrics['reco'], '.4f'), # The reconstruction loss for easier evaluation comparison
        }
        # Optional metrics
        if 'nsdr' in metrics: # Normalized Signal-to-Distortion Ratio
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        # No quantization metrics for now
        if 'grad_norm' in metrics:
            losses['grad_norm'] = format(metrics['grad_norm'], '.4f')
        if 'best' in metrics: # The best performance metric so far
            losses['best'] = format(metrics['best'], '.4f') 
        if 'bname' in metrics: # The name of the best model, either from EMA or from the original model
            losses['bname'] = metrics['bname'] 
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], '.4f')
        if 'hloss' in metrics: # The historical loss metric
            losses['hloss'] = format(metrics['hloss'], '.4f')
        return losses
    
    def _format_test(self, metrics: Dict) -> Dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            # SDR for each source
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            # NSDR for each source
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses
    
    def train(self):
        # In the case of resuming from a previous run (of the same experiment) that is interrupted, but has saved the checkpoint
        # The metrics, learning curves and epoch counting are important to be preserved
        # When loading from (continue_from or continue_pretrained) other checkpoints, the history is not carried over here
        if self.history:
            logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                # Replaying the training metrics
                formatted = self._format_train(metrics['train'])
                logger.info(bold(f'Train Summary ! Epoch {epoch+1}: {_summary(formatted)}'))
                # Replaying the validation metrics
                formatted = self._format_train(metrics['valid'])
                logger.info(bold(f'Valid Summary ! Epoch {epoch+1}: {_summary(formatted)}'))
                # Replaying the test metrics (if any)
                if 'test' in metrics:
                    formatted = self._format_test(metrics['test'])
                    if formatted: # Make sure the test metrics are not empty
                        logger.info(bold(f'Test Summary ! Epoch {epoch+1}: {_summary(formatted)}'))
        
        # Start from the last epoch in the history, and continue on
        for epoch in range(len(self.history), self.args.epochs):
            # Train one epoch
            self.model.train() # Set the model to training mode (which means BatchNorm and Dropout are turned on)
            metrics = {}
            logger.info('-' * 70)
            logger.info("Training...")
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(bold(f'Train Summary ! Epoch {epoch+1}: {_summary(formatted)}'))

            #Validation
            self.model.eval() # Set the model to evaluation mode (which means BatchNorm and Dropout are turned off)
            logger.info('-' * 70)
            logger.info("Validation...")
            with torch.no_grad():
                """
                This with block is used to calculate the validation metrics for the current epoch (main branch, and all EMA models) and their best performance
                *************************************************
                Example of the metrics structure:
                metrics['valid'] = {
                    # Main branch metrics
                    'main': {
                        'reco': 0.123,
                        'loss': 0.456,
                        'nsdr': -5.67,
                        'nsdr_drums': -5.45,
                        'nsdr_bass': -5.89,
                        # ... other metrics
                    },
                    
                    # Individual EMA metrics (like "folders" as you said)
                    'ema_batch_0': {
                        'reco': 0.120,
                        'loss': 0.445,
                        'nsdr': -5.65,
                        'nsdr_drums': -5.43,
                        'nsdr_bass': -5.87,
                        # ... other metrics
                    },
                    'ema_batch_1': {
                        'reco': 0.119,  # <- Let's say this EMA performed best
                        'loss': 0.442,
                        'nsdr': -5.62,
                        'nsdr_drums': -5.41,
                        'nsdr_bass': -5.83,
                        # ... other metrics
                    },
                    
                    # Name of best performing model
                    'bname': 'ema_batch_1',
                    
                    # Best metrics copied at root level via update(bvalid)
                    'reco': 0.119,
                    'loss': 0.442,
                    'nsdr': -5.62,
                    'nsdr_drums': -5.41,
                    'nsdr_bass': -5.83,
                    # ... other metrics
                }
                """
                valid = self._run_one_epoch(epoch, train=False) # A separate data loader is used for validation (when train=False)
                bvalid = valid # The best validation metrics so far
                bname = 'main' # The name of the best model so far
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                key = self.args.test.metric # The metric choosen to be used for validation
                for ema_type, emas in self.emas.items():
                    for k, ema in enumerate(emas): # Any ema is a ModelEMA object
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'{ema_type}_{k}'
                        metrics['valid'][name] = valid
                        a = valid[key]
                        b = bvalid[key]
                        if key.startswith('nsdr'): # For NSDR, the higher the better 
                        # (SDR is not used for training or validation, only at the end for final reporting of the results)
                            a = -a
                            b = -b
                        if a < b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                            state = states.copy_state(self.model.state_dict())
                    metrics['valid'].update(bvalid) # Update the best validation metrics
                    metrics['valid']['bname'] = bname
            
            valid_loss = metrics['valid'][key]
            # A list of all the valid_loss from previous epochs, concatenated with the current epoch's valid_loss
            # (Which is the best valid_loss among the emas/main branch in the current epoch)
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss
            # Only calculate SVD penalty if enabled in args
            if self.args.svd.penalty > 0:
                kw = dict(self.args.svd)
                kw.pop('penalty')
                with torch.no_grad():
                    # Calculate the SVD penalty on the CURRENT model (main branch), not on any EMA models or the best model
                    penalty = svd_penalty(self.model, exact=True, **kw)
                metrics['valid']['penalty'] = penalty
            
            formatted = self._format_train(metrics['valid'])
            logger.info(bold(f'Valid Summary ! Epoch {epoch+1}: {_summary(formatted)}'))

            # Save the best model
            if valid_loss == best_loss or self.args.dset.train_valid: # The train_valid flag is used to save all the model epochs for special cases
                logger.info(bold('New best valid loss %.4f', valid_loss)) # Here cannot use best_loss because maybe not the best model being saved
                self.best_state = states.copy_state(self.model.state_dict())
                self.best_changed = True
            
            # Evaluate model every 'test.every' epochs or on last epoch
            should_eval = (epoch+1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1 
            if should_eval or is_last:
                # Evaluate on the test set
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f'Test Summary ! Epoch {epoch+1}: {_summary(formatted)}'))
            self.link.push_metrics(metrics) # Save the train/valid/test metrics to xp.link

            # Save model each epoch
            self._serialize(epoch)
            logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
            if is_last: # Redundant safety check
                break


            






