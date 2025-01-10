# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs

import logging
import sys
from pathlib import Path
import typing as tp

from dora.log import fatal, bold

from .repo import ModelOnlyRepo, LocalRepo, RemoteRepo, ModelLoadingError

root = Path(__file__).parent.parent
sys.path.append(str(root))
from ClassicalMSS.models.wave_u_net import WaveUNet

# No quantization for now

logger = logging.getLogger(__name__)

SOURCES = ["violin", "piano"]
DEFAULT_MODEL = "wave_u_net"

def wave_u_net_unittest():
    model = WaveUNet()
    return model

def get_model(name:str, repo: tp.Optional[Path] = None): # the '=' sets the default value of repo to 'None'
    """
    The official way of loading a mature model from a repository, with checksum verification,
    potentially from a bag of models or a remote repository.
    """
    if name == 'unittest':
        return wave_u_net_unittest()
    if repo is None:
        fatal("Please specify a repository for the model if you are not using the unittest model. (The remote repository is not supported yet.)")
    if not repo.is_dir():
        fatal(f"{repo} must exist and be a directory.")
    model_repo = LocalRepo(repo)
    model = model_repo.get_model(name)

    model.eval() # set the model to evaluation mode
    return model

def get_model_from_args(args):
    """
    Load local model package or pre-trained model from the arguments.
    """
    if args.name is None:
        args.name = DEFAULT_MODEL
        print(bold("The default model being used now is: "))
    return get_model(args.name, args.repo)