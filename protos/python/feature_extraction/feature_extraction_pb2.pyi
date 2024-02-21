from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ImgBatch(_message.Message):
    __slots__ = ("imgs_pkl", "cam_label_pkl", "view_label_pkl")
    IMGS_PKL_FIELD_NUMBER: _ClassVar[int]
    CAM_LABEL_PKL_FIELD_NUMBER: _ClassVar[int]
    VIEW_LABEL_PKL_FIELD_NUMBER: _ClassVar[int]
    imgs_pkl: bytes
    cam_label_pkl: bytes
    view_label_pkl: bytes
    def __init__(self, imgs_pkl: _Optional[bytes] = ..., cam_label_pkl: _Optional[bytes] = ..., view_label_pkl: _Optional[bytes] = ...) -> None: ...

class FeatureBatch(_message.Message):
    __slots__ = ("features_pkl",)
    FEATURES_PKL_FIELD_NUMBER: _ClassVar[int]
    features_pkl: bytes
    def __init__(self, features_pkl: _Optional[bytes] = ...) -> None: ...
