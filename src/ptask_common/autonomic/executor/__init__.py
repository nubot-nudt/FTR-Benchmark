from abc import abstractmethod
from ptask_common.utils.patterns import singleton, BaseFactory


class BaseExecutor:

    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


@singleton
class ExecutorFactory(BaseFactory):

    def __init__(self):
        super().__init__(parent_cls=BaseExecutor)
