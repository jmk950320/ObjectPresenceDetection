import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


class MaskProcessor:
    """
    마스크 영역을 처리하는 클래스
    """
    
    @staticmethod
    def load_mask(mask_path: str) -> np.ndarray:
        """
        마스크 파일을 로드
        
        Args:
            mask_path (str): 마스크 파일 경로
        
        Returns:
            np.ndarray: 마스크 배열 (0-255 또는 0-1)
        """
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"마스크 파일을 찾을 수 없습니다: {mask_path}")
        
        # 마스크 로드 (그레이스케일)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"마스크 파일을 읽을 수 없습니다: {mask_path}")
        
        return mask
    
    @staticmethod
    def apply_mask(frame: np.ndarray, mask: np.ndarray, mask_value: int = 255) -> Tuple[np.ndarray, np.ndarray]:
        """
        프레임에 마스크를 적용
        
        Args:
            frame (np.ndarray): 원본 프레임 (RGB)
            mask (np.ndarray): 마스크 배열
            mask_value (int): 마스크에서 유효한 영역의 값 (기본값: 255)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (마스크된 프레임, 이진 마스크)
        """
        # 마스크 크기를 프레임 크기에 맞게 조정
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # 이진 마스크 생성
        binary_mask = (mask >= mask_value).astype(np.uint8)
        
        # 3채널로 확장
        if len(frame.shape) == 3:
            mask_3d = np.stack([binary_mask] * frame.shape[2], axis=2)
        else:
            mask_3d = binary_mask
        
        # 마스크 적용
        masked_frame = frame * mask_3d
        
        return masked_frame, binary_mask
    
    @staticmethod
    def extract_masked_region(frame: np.ndarray, mask: np.ndarray, mask_value: int = 255) -> np.ndarray:
        """
        마스크 영역만 추출 (배경을 투명하게)
        
        Args:
            frame (np.ndarray): 원본 프레임 (RGB)
            mask (np.ndarray): 마스크 배열
            mask_value (int): 마스크에서 유효한 영역의 값
        
        Returns:
            np.ndarray: 마스크된 영역만 포함한 프레임 (RGBA)
        """
        masked_frame, binary_mask = MaskProcessor.apply_mask(frame, mask, mask_value)
        
        # RGBA로 변환 (알파 채널 추가)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgba_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=frame.dtype)
            rgba_frame[:, :, :3] = masked_frame
            rgba_frame[:, :, 3] = binary_mask * 255  # 알파 채널
        else:
            rgba_frame = masked_frame
        
        return rgba_frame
    
    @staticmethod
    def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        마스크 크기 조정
        
        Args:
            mask (np.ndarray): 원본 마스크
            target_size (Tuple[int, int]): 목표 크기 (width, height)
        
        Returns:
            np.ndarray: 크기 조정된 마스크
        """
        return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    @staticmethod
    def save_mask(mask: np.ndarray, output_path: str):
        """
        마스크를 파일로 저장
        
        Args:
            mask (np.ndarray): 저장할 마스크
            output_path (str): 출력 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask)
    
    @staticmethod
    def get_mask_statistics(mask: np.ndarray, mask_value: int = 255) -> dict:
        """
        마스크 통계 정보 반환
        
        Args:
            mask (np.ndarray): 마스크 배열
            mask_value (int): 마스크 유효 영역 값
        
        Returns:
            dict: 마스크 통계 (총 픽셀 수, 마스크 픽셀 수, 비율 등)
        """
        binary_mask = (mask >= mask_value).astype(np.uint8)
        total_pixels = mask.shape[0] * mask.shape[1]
        masked_pixels = np.sum(binary_mask)
        mask_ratio = masked_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'total_pixels': total_pixels,
            'masked_pixels': int(masked_pixels),
            'mask_ratio': float(mask_ratio),
            'mask_percentage': float(mask_ratio * 100),
            'shape': mask.shape
        }
    
    @staticmethod
    def combine_masks(masks: list, operation: str = 'union') -> np.ndarray:
        """
        여러 마스크를 결합
        
        Args:
            masks (list): 마스크 배열 리스트
            operation (str): 결합 방식 ('union', 'intersection', 'difference')
        
        Returns:
            np.ndarray: 결합된 마스크
        """
        if not masks:
            raise ValueError("마스크 리스트가 비어있습니다")
        
        result = masks[0].copy()
        
        for mask in masks[1:]:
            if mask.shape != result.shape:
                mask = cv2.resize(mask, (result.shape[1], result.shape[0]))
            
            if operation == 'union':
                result = cv2.bitwise_or(result, mask)
            elif operation == 'intersection':
                result = cv2.bitwise_and(result, mask)
            elif operation == 'difference':
                result = cv2.bitwise_xor(result, mask)
            else:
                raise ValueError(f"지원하지 않는 연산: {operation}")
        
        return result
    
    @staticmethod
    def invert_mask(mask: np.ndarray) -> np.ndarray:
        """
        마스크 반전 (foreground <-> background)
        
        Args:
            mask (np.ndarray): 원본 마스크
        
        Returns:
            np.ndarray: 반전된 마스크
        """
        return cv2.bitwise_not(mask)
    
    @staticmethod
    def dilate_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """
        마스크 팽창 (확장)
        
        Args:
            mask (np.ndarray): 원본 마스크
            kernel_size (int): 커널 크기
            iterations (int): 반복 횟수
        
        Returns:
            np.ndarray: 팽창된 마스크
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=iterations)
    
    @staticmethod
    def erode_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """
        마스크 침식 (축소)
        
        Args:
            mask (np.ndarray): 원본 마스크
            kernel_size (int): 커널 크기
            iterations (int): 반복 횟수
        
        Returns:
            np.ndarray: 침식된 마스크
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(mask, kernel, iterations=iterations)
    
    @staticmethod
    def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        마스크 경계 부드럽게 처리
        
        Args:
            mask (np.ndarray): 원본 마스크
            kernel_size (int): 블러 커널 크기 (홀수여야 함)
        
        Returns:
            np.ndarray: 부드러워진 마스크
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)


if __name__ == "__main__":
    # 사용 예시
    try:
        # 마스크 로드
        mask = MaskProcessor.load_mask("path/to/mask.png")
        
        # 마스크 통계
        stats = MaskProcessor.get_mask_statistics(mask)
        print(f"마스크 통계: {stats}")
        
        # 마스크 처리
        dilated = MaskProcessor.dilate_mask(mask, kernel_size=5)
        smoothed = MaskProcessor.smooth_mask(mask, kernel_size=7)
        
        print("마스크 처리 완료")
        
    except FileNotFoundError as e:
        print(f"파일 오류: {e}")
    except Exception as e:
        print(f"처리 오류: {e}")
