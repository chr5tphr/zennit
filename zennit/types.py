'''Type definitions for convenience.'''
import torch


class SubclassMeta(type):
    '''Meta class to bundle multiple subclasses.'''
    def __instancecheck__(cls, inst):
        """Implement isinstance(inst, cls) as subclasscheck."""
        return cls.__subclasscheck__(type(inst))

    def __subclasscheck__(cls, sub):
        """Implement issubclass(sub, cls) with by considering additional __subclass__ members."""
        candidates = cls.__dict__.get("__subclass__", tuple())
        return type.__subclasscheck__(cls, sub) or issubclass(sub, candidates)


class Convolution(metaclass=SubclassMeta):
    '''Abstract base class that describes convolutional modules.'''
    __subclass__ = (
        torch.nn.modules.conv.Conv1d,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.conv.Conv3d,
        torch.nn.modules.conv.ConvTranspose1d,
        torch.nn.modules.conv.ConvTranspose2d,
        torch.nn.modules.conv.ConvTranspose3d,
    )


class Linear(metaclass=SubclassMeta):
    '''Abstract base class that describes linear modules.'''
    __subclass__ = (
        Convolution,
        torch.nn.modules.linear.Linear,
    )


class BatchNorm(metaclass=SubclassMeta):
    '''Abstract base class that describes batch normalization modules.'''
    __subclass__ = (
        torch.nn.modules.batchnorm.BatchNorm1d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.batchnorm.BatchNorm3d,
    )
