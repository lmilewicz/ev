from abc import ABCMeta, abstractmethod

class Module(object, metaclass=ABCMeta):
    def __init__(self, layerType):
            self.layerType = layerType

    # @abstractmethod
    # def create(self):
    #     raise NotImplementedError("Subclass of Node does not implement create_node()'")
