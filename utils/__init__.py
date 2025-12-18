"""
Video and Mask Processing Utilities

이 패키지는 비디오 처리와 마스크 처리를 위한 유틸리티들을 제공합니다.
"""

from .mask_processor import MaskProcessor
from .read_video import VideoFrameReader, VideoMaskProcessor

__all__ = [
    'MaskProcessor',
    'VideoFrameReader', 
    'VideoMaskProcessor'
]

__version__ = '1.0.0'