# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs

from contextlib import contextmanager

import warnings
import functools
import hashlib
import io
from pathlib import Path
import inspect

from omegaconf import OmegaConf # A YAML-based hierarchical configuration system
import torch

# ----------------------------------------------------------------------------------------------------
# Utility functions (no quantization for now)

def capture_init(init):
    """
    This decorator is used on the __init__ method of a class to capture the arguments passed to the constructor.
    It adds a saving method before the original constructor is called.
    """
    @functools.wraps(init) # This decorator is used to preserve the metadata of the original function (e.g. name, docstring, etc.)
    def new_init(self, *args, **kwargs):
        self.init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs) # The original constructor
    return new_init

def save_with_checksum(content, path):
    """
    Save the given value on disk, along with a sha256 hash.
    The checksum verifies the integrity of the file, and the filename is automatically versioned.
    The content is saved on the buffer in memory, then written to disk to accelerate the process compared to saving the content twice on disk.
    Should be used with the output of either `serialize_model` or `get_state`.
    """
    buffer = io.BytesIO() # Create a buffer in memory
    torch.save(content, buffer) # Save the content to the buffer (which is faster than saving to disk)
    checksum_signature = hashlib.sha256(buffer.getvalue()).hexdigest()[:8] # Compute the sha256 hash of the content

    path = path.parent / (path.stem + "-" + checksum_signature + path.suffix) # path.stem: The name of the file without the extension
    path.write_bytes(buffer.getvalue()) # Write the content to disk

# ----------------------------------------------------------------------------------------------------
# The following functions are used to copy, get, set the state of a model
# and swap a new state (like an EMA state) with the current state of a model temporarily
# to perform tests or evaluations

def copy_state(state):
    """
    Copy the state of a model.
    """
    return {name: tensor.cpu().clone() for name, tensor in state.items()}

def get_state(model, half=False):
    """
    Get the state of a model, no quantization for now.
    If half is True, the state is stored as half precision (FP16) which half the state size
    with a very small probability of performance degradation.
    """
    dtype = torch.half if half else None
    # .data is used to get the data of the tensor without the gradient, which will also be disconnected from the computational graph
    state = {name: tensor.data.to(device='cpu', dtype=dtype) for name, tensor in model.state_dict().items()}
    return state

def set_state(model, state):
    """
    Set the state on a given model, no quantization for now.
    """
    model.load_state_dict(state, strict=False)
    return state

@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state, we can do tests or evaluate metrics on this new state
        # model back to old state
    """
    # Since here the swap_state is a context manager, it should be called with a with statement
    # the code before "yield" keyword will be executed first,
    # and then the code in the outer file within the with block will be executed
    # after the with block is finished, the code after "yield" keyword will be executed (cleaning up)
    # Here this "cleaning up" is further ensured by the finally clause which will be executed regardless of how the with block is exited normally.
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)

# ----------------------------------------------------------------------------------------------------
# The following function is used to entirely serialize (save) a model
# (including the model's state, meta-arguments, meta-keyword-arguments, training arguments, etc.)
# so that it can be restored altogether later in its entirety

def serialize_model(model, training_args, half=True):
    """
    Serialize (save) the model state and training arguments so that it can be restored altogether later using the load_model function.
    No quantization for now.
    """

    args, kwargs = model.init_args_kwargs # The model's meta-arguments
    training_args = OmegaConf.to_container(training_args, resolve=True) # The training arguments, all interpolations (insertions) are resolved
    # There are two types of args here:
    # 1. The meta-arguments of the model, which are the arguments passed to the constructor of the model, such as:
    #    - The parameters of each layer/module (kernel size, stride, padding, etc.)
    # 2. The training arguments that facilitate the training process (which are not model-specific), such as:
    #    - The learning rate, batch size, number of epochs, 
    #    - optimizer settings, loss function, etc.

    klass = model.__class__ # object.__class__: Get the class (type) of the object, here essentially what model this is

    state = get_state(model, half) # The state of the model, no quantization for now

    return {
        'klass': klass, # The keyword 'class' is reserved in Python, so we use 'klass' instead
        'args': args, # The meta-arguments of the model
        'kwargs': kwargs, # The meta-keyword-arguments of the model
        'state': state, # The current state of the model
        'training_args': training_args, # The training arguments
    }

def load_model(path_or_package, strict=False):
    """
    Load a model from a given serialized model, either given as a dict (package, already loaded)
    or as a path to a file on disk.
    """
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)): # if it is either a string or a Path object
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, 'cpu')
    else:
        raise ValueError(f"Invalid input type for load_model: {path_or_package} is of the type {type(path_or_package)} while it should be a dict, str, or a Path object")

    klass = package['klass']
    args = package['args']
    kwargs = package['kwargs']
    
    if strict:
        model = klass(*args, **kwargs)
    else: # Handles backward compatibility with old models
        constructor_parameters = inspect.signature(klass) # inspect.signature(func): Get the signature (constructor in the case of a class) of a function
        for key in list(kwargs):
            if key not in constructor_parameters.parameters:
                warnings.warn("Dropping inexistent parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)
    
    state = package['state']

    set_state(model, state)
    return model


