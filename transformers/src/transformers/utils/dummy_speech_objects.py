# This file is autogenerated by the command `make fix-copies`, do not edit.
# flake8: noqa
from ..utils import DummyObject, requires_backends


class Speech2TextFeatureExtractor(metaclass=DummyObject):
    _backends = ["speech"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["speech"])
