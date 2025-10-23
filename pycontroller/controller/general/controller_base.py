from abc import ABC, abstractmethod


class ControllerBase(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, state, target, dt: float):
        pass
