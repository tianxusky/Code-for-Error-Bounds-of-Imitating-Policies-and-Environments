# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import abc
from typing import Union
import lunzi.nn as nn


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def get_next_states(self, states, actions):
        pass


BaseNNPolicy = Union[BasePolicy, nn.Module]  # should be Intersection, see PEP544
