import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Union
import os
try:
    from .mask_processor import MaskProcessor
except ImportError:
    # 직접 실행할 때를 위한 fallback
    from mask_processor import MaskProcessor

class VideoFrameReader:
    """
    비디오를 프레임별로 읽고 마스크 영역을 처리하는 클래스
    """
    
    def __init__(self, video_path: str):
        """
        비디오 리더 초기화
        
        Args:
            video_path (str): 비디오 파일 경로
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 비디오 정보 가져오기
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def release(self):
        """비디오 캡처 객체 해제"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
    
    def get_video_info(self) -> dict:
        """
        비디오 정보 반환
        
        Returns:
            dict: 비디오 정보 (fps, frame_count, width, height)
        """
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0
        }
    
    def read_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        비디오 프레임을 순차적으로 읽어오는 제너레이터
        
        Args:
            start_frame (int): 시작 프레임 번호 (기본값: 0)
            end_frame (Optional[int]): 종료 프레임 번호 (기본값: None, 마지막까지)
        
        Yields:
            Tuple[int, np.ndarray]: (프레임 번호, 프레임 이미지)
        """
        if end_frame is None:
            end_frame = self.frame_count
        
        # 시작 프레임으로 이동
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield current_frame, frame_rgb
            current_frame += 1
    
    def read_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        특정 프레임 번호의 프레임을 읽어옴
        
        Args:
            frame_number (int): 읽어올 프레임 번호
        
        Returns:
            Optional[np.ndarray]: 프레임 이미지 (RGB), 실패시 None
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None





class VideoMaskProcessor:
    """
    비디오와 마스크를 함께 처리하는 통합 클래스
    """
    
    def __init__(self, video_path: str, mask_path: Optional[str] = None):
        """
        비디오-마스크 프로세서 초기화
        
        Args:
            video_path (str): 비디오 파일 경로
            mask_path (Optional[str]): 마스크 파일 경로 (선택사항)
        """
        self.video_reader = VideoFrameReader(video_path)
        self.mask = None
        
        if mask_path:
            self.load_mask(mask_path)
    
    def load_mask(self, mask_path: str):
        """
        마스크 파일 로드
        
        Args:
            mask_path (str): 마스크 파일 경로
        """
        self.mask = MaskProcessor.load_mask(mask_path)
    
    def process_frames_with_mask(self, 
                               start_frame: int = 0, 
                               end_frame: Optional[int] = None,
                               mask_value: int = 255) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        비디오 프레임을 읽으면서 마스크를 적용
        
        Args:
            start_frame (int): 시작 프레임 번호
            end_frame (Optional[int]): 종료 프레임 번호
            mask_value (int): 마스크 유효 영역 값
        
        Yields:
            Tuple[int, np.ndarray, np.ndarray]: (프레임 번호, 원본 프레임, 마스크된 프레임)
        """
        if self.mask is None:
            raise ValueError("마스크가 로드되지 않았습니다. load_mask()를 먼저 호출하세요.")
        
        for frame_num, frame in self.video_reader.read_frames(start_frame, end_frame):
            masked_frame, _ = MaskProcessor.apply_mask(frame, self.mask, mask_value)
            yield frame_num, frame, masked_frame
    
    def get_video_info(self) -> dict:
        """비디오 정보 반환"""
        return self.video_reader.get_video_info()
    
    def release(self):
        """리소스 해제"""
        self.video_reader.release()
    
    def __enter__(self):
        return self
    
        self.release()


class VideoROIProcessor:
    """
    비디오에서 ROI 영역만 크롭하여 처리하는 클래스
    """
    
    def __init__(self, video_path: str, roi: Tuple[int, int, int, int], roi_format: str = 'xywh'):
        """
        비디오-ROI 프로세서 초기화
        
        Args:
            video_path (str): 비디오 파일 경로
            roi (Tuple[int, int, int, int]): ROI 좌표
            roi_format (str): ROI 형식 ('xywh' 또는 'xyxy')
                - 'xywh': (x, y, width, height)
                - 'xyxy': (x1, y1, x2, y2)
        """
        self.video_reader = VideoFrameReader(video_path)
        self.roi_format = roi_format
        
        # ROI 좌표를 (x, y, w, h) 형식으로 변환
        if roi_format == 'xywh':
            self.x, self.y, self.w, self.h = roi
        elif roi_format == 'xyxy':
            x1, y1, x2, y2 = roi
            self.x, self.y = x1, y1
            self.w, self.h = x2 - x1, y2 - y1
        else:
            raise ValueError(f"지원하지 않는 ROI 형식: {roi_format}. 'xywh' 또는 'xyxy'를 사용하세요.")
        
        # ROI 유효성 검사
        video_info = self.video_reader.get_video_info()
        if self.x < 0 or self.y < 0 or self.w <= 0 or self.h <= 0:
            raise ValueError(f"잘못된 ROI 좌표: ({self.x}, {self.y}, {self.w}, {self.h})")
        if self.x + self.w > video_info['width'] or self.y + self.h > video_info['height']:
            raise ValueError(f"ROI가 비디오 범위를 벗어남: ROI=({self.x}, {self.y}, {self.w}, {self.h}), Video=({video_info['width']}, {video_info['height']})")
    
    def process_frames_with_roi(self,
                               start_frame: int = 0,
                               end_frame: Optional[int] = None) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        비디오 프레임을 읽으면서 ROI 영역만 크롭
        
        Args:
            start_frame (int): 시작 프레임 번호
            end_frame (Optional[int]): 종료 프레임 번호
        
        Yields:
            Tuple[int, np.ndarray, np.ndarray]: (프레임 번호, 원본 프레임, ROI 크롭된 프레임)
        """
        for frame_num, frame in self.video_reader.read_frames(start_frame, end_frame):
            # ROI 영역 크롭
            cropped_frame = frame[self.y:self.y+self.h, self.x:self.x+self.w]
            yield frame_num, frame, cropped_frame
    
    def get_video_info(self) -> dict:
        """비디오 정보 반환"""
        info = self.video_reader.get_video_info()
        info['roi'] = (self.x, self.y, self.w, self.h)
        info['roi_format'] = self.roi_format
        return info
    
    def release(self):
        """리소스 해제"""
        self.video_reader.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoMaskROIProcessor:
    """
    비디오에 마스크를 적용한 후 ROI 영역만 크롭하여 처리하는 클래스
    """
    
    def __init__(self, video_path: str, mask_path: str, roi: Tuple[int, int, int, int], roi_format: str = 'xywh'):
        """
        비디오-마스크-ROI 프로세서 초기화
        
        Args:
            video_path (str): 비디오 파일 경로
            mask_path (str): 마스크 파일 경로
            roi (Tuple[int, int, int, int]): ROI 좌표
            roi_format (str): ROI 형식 ('xywh' 또는 'xyxy')
        """
        self.video_reader = VideoFrameReader(video_path)
        self.mask = MaskProcessor.load_mask(mask_path)
        self.roi_format = roi_format
        
        # ROI 좌표를 (x, y, w, h) 형식으로 변환
        if roi_format == 'xywh':
            self.x, self.y, self.w, self.h = roi
        elif roi_format == 'xyxy':
            x1, y1, x2, y2 = roi
            self.x, self.y = x1, y1
            self.w, self.h = x2 - x1, y2 - y1
        else:
            raise ValueError(f"지원하지 않는 ROI 형식: {roi_format}. 'xywh' 또는 'xyxy'를 사용하세요.")
        
        # ROI 유효성 검사
        video_info = self.video_reader.get_video_info()
        if self.x < 0 or self.y < 0 or self.w <= 0 or self.h <= 0:
            raise ValueError(f"잘못된 ROI 좌표: ({self.x}, {self.y}, {self.w}, {self.h})")
        if self.x + self.w > video_info['width'] or self.y + self.h > video_info['height']:
            raise ValueError(f"ROI가 비디오 범위를 벗어남: ROI=({self.x}, {self.y}, {self.w}, {self.h}), Video=({video_info['width']}, {video_info['height']})")
    
    def process_frames_with_mask_and_roi(self,
                                        start_frame: int = 0,
                                        end_frame: Optional[int] = None,
                                        mask_value: int = 255) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        비디오 프레임을 읽으면서 마스크를 적용한 후 ROI 영역만 크롭
        
        Args:
            start_frame (int): 시작 프레임 번호
            end_frame (Optional[int]): 종료 프레임 번호
            mask_value (int): 마스크 유효 영역 값
        
        Yields:
            Tuple[int, np.ndarray, np.ndarray]: (프레임 번호, 원본 프레임, 마스크+ROI 처리된 프레임)
        """
        for frame_num, frame in self.video_reader.read_frames(start_frame, end_frame):
            # 1. 마스크 적용
            masked_frame, _ = MaskProcessor.apply_mask(frame, self.mask, mask_value)
            
            # 2. ROI 영역 크롭
            cropped_frame = masked_frame[self.y:self.y+self.h, self.x:self.x+self.w]
            
            yield frame_num, frame, cropped_frame
    
    def get_video_info(self) -> dict:
        """비디오 정보 반환"""
        info = self.video_reader.get_video_info()
        info['roi'] = (self.x, self.y, self.w, self.h)
        info['roi_format'] = self.roi_format
        info['has_mask'] = True
        return info
    
    def release(self):
        """리소스 해제"""
        self.video_reader.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()



# 사용 예시 함수들
def process_video_with_mask_example(video_path: str, mask_path: str, output_dir: str = "./output"):
    """
    비디오와 마스크를 처리하는 예시 함수
    
    Args:
        video_path (str): 비디오 파일 경로
        mask_path (str): 마스크 파일 경로
        output_dir (str): 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with VideoMaskProcessor(video_path, mask_path) as processor:
        video_info = processor.get_video_info()
        print(f"비디오 정보: {video_info}")
        
        # 처음 10프레임만 처리
        for frame_num, original_frame, masked_frame in processor.process_frames_with_mask(0, 10):
            # 원본 프레임 저장
            original_path = output_path / f"frame_{frame_num:04d}_original.jpg"
            cv2.imwrite(str(original_path), cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
            
            # 마스크된 프레임 저장
            masked_path = output_path / f"frame_{frame_num:04d}_masked.jpg"
            cv2.imwrite(str(masked_path), cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR))
            
            print(f"프레임 {frame_num} 처리 완료")


def read_video(video_path: str):
    """
    OpenCV를 사용하여 비디오를 읽고 프레임을 하나씩 반환하는 제너레이터
    
    Args:
        video_path (str): 비디오 파일 경로
        
    Yields:
        np.ndarray: RGB 포맷의 프레임 이미지
    """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 비디오를 열 수 없습니다 - {video_path}")
        return

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            # 더 이상 프레임이 없거나 읽기 실패시 종료
            if not ret:
                break
                
            # OpenCV는 기본적으로 BGR로 읽으므로, RGB로 변환하여 반환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb
            
    finally:
        # 자원 해제
        cap.release()


if __name__ == "__main__":
    # 사용 예시
    video_path = "path/to/your/video.mp4"
    mask_path = "path/to/your/mask.png"
    
    # 기본 사용법
    try:
        with VideoMaskProcessor(video_path, mask_path) as processor:
            info = processor.get_video_info()
            print(f"비디오 정보: {info}")
            
            # 첫 번째 프레임 처리
            for frame_num, original, masked in processor.process_frames_with_mask(0, 1):
                print(f"프레임 {frame_num}: 원본 크기 {original.shape}, 마스크된 크기 {masked.shape}")
                break
                
    except FileNotFoundError as e:
        print(f"파일 오류: {e}")
    except Exception as e:
        print(f"처리 오류: {e}")