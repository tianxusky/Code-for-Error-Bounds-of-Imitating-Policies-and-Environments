# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import abc
from typing import Union
import lunzi.nn as nn


class BaseVFunction(abc.ABC):
    @abc.abstractmethod
    def get_values(self, states, actions):
        pass


BaseNNVFunction = Union[BaseVFunction, nn.Module]  # in fact it should be Intersection