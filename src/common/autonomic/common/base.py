
from abc import abstractmethod

class BasePredictor():

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class BaseExecutor():

    @abstractmethod
    def get_action(self, obs):
        pass


    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)