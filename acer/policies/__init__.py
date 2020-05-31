import abc
from typing import Union, List
import lunzi.nn as nn


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, states):
        pass

    @abc.abstractmethod
    def get_q_values(self, states, actions_):
        pass

    @abc.abstractmethod
    def get_v_values(self, states):
        pass


BaseNNPolicy = Union[BasePolicy, nn.Module]  # should be Intersection, see PEP544
