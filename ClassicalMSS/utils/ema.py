# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs
# Inspired from https://github.com/rwightman/pytorch-image-models

from contextlib import contextmanager
import torch

from .states import swap_state

class ModelEMA:
    """
    Perform EMA (Exponential Moving Average) on a model. You can switch to the EMA weights temporarily
    with the `swap` method.

        ema = ModelEMA(model)
        with ema.swap():
            # compute valid metrics with averaged model.
    """
    def __init__(self, model, decay=0.9999, unbias=True, device='cpu'):
        self.decay = decay
        self.model = model
        self.state = {}
        self.count = 0
        self.device = device
        self.unbias = unbias

        self._init()

    def _init(self):
        """
        This initialization is to copy all of the float32 parameters and the names
        to a new dictionary (self.state) which is disconnected from the computational graph
        """
        for name, tensor in self.model.state_dict().items():
            # Ignore non-float32 tensors for they may be boolean flags or integer indices
            if tensor.dtype != torch.float32:
                continue
            device = self.device or tensor.device # If self.device is None, use the device of the tensor (model parameters)
            self.state[name] = torch.zeros_like(tensor, device=device)
            if name not in self.state:
                self.state[name] = tensor.detach().to(device, copy=True) # disconnected from the computational graph
        
    def update(self):
        # --------------------------------------------------------------------------------------------------------------------------
        # The biased EMA is the standard EMA, it is biased because: (if the decay is 0.9999)
        # The first period parameter A_0 will be just the parameter itself
        # The second period parameter A_1_EMA will be 0.9999 * A_0 + 0.0001 * A_1 which strongly biased towards A_0, the first parameter
        # --------------------------------------------------------------------------------------------------------------------------
        # There is more than one way to try to fix this bias
        # In the current way, if decay = 1, then the EMA will be exactly the arithmetic mean of all periods' parameters
        # if decay = 0, w will always be 1, so the EMA will be the last period's parameter, that is the same as the biased EMA case
        # The smaller the decay, the more emphasis on the last period's parameter
        # --------------------------------------------------------------------------------------------------------------------------
        if self.unbias:
            self.count = self.count * self.decay + 1
            w = 1 / self.count
        else:
            w = 1 - self.decay

        for name, tensor in self.model.state_dict().items():
            if tensor.dtype != torch.float32:
                continue
            device = self.device or tensor.device
            self.state[name].mul_(1 - w).add_(tensor.detach().to(device), alpha=w)
    
    @contextmanager
    def swap(self):
        """
        This method is to swap the state of the model to the EMA state, so that we can do tests or evaluate metrics on the EMA state
        """
        # The swap_state method itself is a context manager, so it should be called with a with statement
        # See the .states.py file for more details (and error handling)
        with swap_state(self.model, self.state):
            yield
    
    def state_dict(self):
        """
        This method is to return the state of the EMA parameters,
        the 'state' is the EMA state_dict, and the count is the number of updates (averages) happened (which is only true when unbias is True)
        """
        return {'state': self.state, 'count': self.count}

    def load_state_dict(self, state):
        """
        Loading the state of the EMA parameters saved by this ModelEMA class
        (as the general state_dict may not have the 'count' key)
        """
        self.count = state['count']
        for name, tensor in state['state'].items():
            self.state[name].copy_(tensor)


