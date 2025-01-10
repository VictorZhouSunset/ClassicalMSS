# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs
"""Represents a model repository, including pre-trained models and bags of models.
No remote repository or bags of models for now. For now, all repositories are local and model-only.
"""

from pathlib import Path
from hashlib import sha256
import typing as tp

from abc import ABC, abstractmethod # Abstract Class

import torch
import yaml

from .states import load_model
from .apply import Model
# No bags of models for now, will be implemented later
# No remote repository for now, may be implemented later

class ModelLoadingError(RuntimeError):
    """Distinguish between model loading failures and other runtime errors."""
    pass

def check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, "rb") as file:
        while True:
            buffer = file.read(2**20) # Read in chunks of 2^20 (1024^2) bytes which is 1MB
            if not buffer:
                break
            sha.update(buffer) # Because SHA-256 is a cryptographic hash function that is deterministic (or incremental),
                                # we can update the hash partial by partial without having to read the entire file into memory
    actual_checksum = sha.hexdigest()[:len(checksum)] # The checksum for comparison is a truncated checksum of the file
    if actual_checksum != checksum:
        raise ModelLoadingError(f"Invalid checksum for file {path}. Expected {checksum}, got {actual_checksum}.")

class ModelOnlyRepo(ABC):
    """Abstract base class for all model only repositories. (In contrast to bags of models)"""

    @abstractmethod
    def has_model(self, signature: str) -> bool:
        pass

    @abstractmethod
    def load_model(self, signature: str) -> Model:
        pass

    @abstractmethod
    def list_models(self) -> tp.Dict[str, tp.Union[str, Path]]:
        pass

class LocalRepo(ModelOnlyRepo):
    """A local repository of models. The input is a Path object that contains (potentially multiple) pre-trained models."""

    def __init__(self, root: Path):
        self.root = root
        self.scan()

    def scan(self):
        self._models = {} # Dictionary of models: {xp_signature: file_path}
        self._checksums = {} # Dictionary of checksums: {xp_signature: checksum}
        for file in self.root.iterdir(): # returns an iterator of all files and directories (as Path objects) contained in a directory
            if file.suffix == ".th":
                if '-' in file.stem:
                    xp_signature, checksum = file.stem.split('-')
                    self._checksums[xp_signature] = checksum
                else:
                    xp_signature = file.stem
                if xp_signature in self._models:
                    raise ModelLoadingError(f'Duplicate pre-trained model exist for signature {xp_signature}'
                                            f' found in {self.root}. Please delete all but one.')
                self._models[xp_signature] = file

    def has_model(self, signature: str) -> bool:
        return signature in self._models

    def get_model(self, signature: str) -> Model:
        """
        Load a pre-trained model from the repository with the load_model function
        A checksum is also performed before loading the model to ensure the integrity of the model.
        """
        try:
            file = self._models[signature]
        except KeyError:
            raise ModelLoadingError(f'Could not find a pre-trained model with signature {signature} in {self.root}.')
        if signature in self._checksums:
            check_checksum(file, self._checksums[signature])
        return load_model(file)
    
    def list_models(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return self._models

class RemoteRepo(ModelOnlyRepo):
    """A remote repository of models. The input is a dictionary of (potentially multiple) pre-trained models: {signature: URL}."""

    def __init__(self, models: tp.Dict[str, str]):
        self._models = models # Dictionary of models: {signature: URL}

    def has_model(self, signature: str) -> bool:
        return signature in self._models

    def get_model(self, signature: str) -> Model:
        try:
            url = self._models[signature]
        except KeyError:
            raise ModelLoadingError(f'Could not find a pre-trained model with signature {signature} in {self.root}.')
        # torch.hub provides a simple way to load a pre-trained model (the its weights) from a URL or an online repository (like GitHub)
        package = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
        # The model file should be companioned with a <filename>.hash file which will be used for the check_hash option
        
        return load_model(package)

    def list_models(self) -> tp.Dict[str, str]:
        return self._models


