from abc import ABC, abstractmethod


class ControllerBase(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, measurement, target, dt: float):
        pass
