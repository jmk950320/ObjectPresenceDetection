"""
ResNet, MobileNet, VGG 등에서 특징 추출하는 간단한 인터페이스

기존 코드와 동일한 방식으로 사용할 수 있도록 설계됨:
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)  # feature cnt, dims
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from feature_extractor import FeatureExtractor
from utils.read_video import read_video

class UnifiedCNNModel:
    """
    ResNet, MobileNet, VGG 등을 통합한 모델 래퍼
    기존 코드와 동일한 인터페이스 제공
    """
    
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True, device: str = 'cuda'):
        """
        모델 초기화
        
        Args:
            model_name (str): 모델 이름 ('resnet50', 'resnet101', 'mobilenet_v2', 'vgg16', 'vgg19')
            pretrained (bool): 사전 학습된 가중치 사용 여부
            device (str): 디바이스 ('cuda' 또는 'cpu')
        """
        self.extractor = FeatureExtractor(model_name, pretrained, device)
        self.model_name = model_name
        self.device = device
    
    def get_intermediate_layers(
        self, 
        image: torch.Tensor, 
        n: Optional[Union[List[int], range]] = None,
        reshape: bool = True,
        norm: bool = True
    ) -> List[torch.Tensor]:
        """
        중간 레이어 특징 추출 (기존 인터페이스와 호환)
        
        Args:
            image (torch.Tensor): 입력 이미지 (B, C, H, W)
            n (Optional[Union[List[int], range]]): 추출할 레이어 인덱스
            reshape (bool): 특징을 reshape 할지 여부
            norm (bool): 정규화 여부
        
        Returns:
            List[torch.Tensor]: 추출된 특징 리스트
        """
        # n이 range 객체인 경우 리스트로 변환
        if isinstance(n, range):
            layer_indices = list(n)
        elif n is None:
            layer_indices = None
        else:
            layer_indices = n
        
        # 특징 추출
        features = self.extractor.extract_features(
            image, 
            layer_indices=layer_indices,
            normalize=norm,
            reshape=reshape
        )
        
        return features
    
    def cuda(self):
        """CUDA로 이동 (호환성을 위한 메서드)"""
        self.extractor.device = 'cuda'
        self.extractor.model = self.extractor.model.cuda()
        return self
    
    def cpu(self):
        """CPU로 이동 (호환성을 위한 메서드)"""
        self.extractor.device = 'cpu'
        self.extractor.model = self.extractor.model.cpu()
        return self
    
    def eval(self):
        """평가 모드로 설정 (호환성을 위한 메서드)"""
        self.extractor.model.eval()
        return self


def create_model(model_name: str = 'resnet50', pretrained: bool = True, device: str = 'cuda') -> UnifiedCNNModel:
    """
    통합 모델 생성 함수
    
    Args:
        model_name (str): 모델 이름
        pretrained (bool): 사전 학습된 가중치 사용 여부
        device (str): 디바이스
    
    Returns:
        UnifiedCNNModel: 통합 모델 객체
    """
    return UnifiedCNNModel(model_name, pretrained, device)


def extract_features_like_original(
    image_resized_norm: torch.Tensor,
    model_name: str = 'resnet50',
    n_layers: Union[int, List[int]] = 4,
    device: str = 'cuda',
    model: Optional[UnifiedCNNModel] = None
) -> torch.Tensor:
    """
    기존 코드와 동일한 방식으로 특징 추출
    
    Args:
        image_resized_norm (torch.Tensor): 정규화된 이미지 (C, H, W)
        model_name (str): 모델 이름
        n_layers (Union[int, List[int]]): 추출할 레이어 수 또는 특정 레이어 인덱스 리스트
                                          예: 4 -> [0,1,2,3], [2,3] -> 2번과 3번 레이어만
        device (str): 디바이스
        model (Optional[UnifiedCNNModel]): 미리 로드된 모델 객체 (None이면 새로 생성)
    
    Returns:
        torch.Tensor: 추출된 특징 (feature_cnt, dims)
    """
    # 모델 생성 (전달받지 않은 경우에만)
    if model is None:
        model = create_model(model_name, pretrained=True, device=device)
    
    # 기존 코드와 동일한 방식으로 특징 추출
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.float32):
            # n_layers가 int면 range로, list면 그대로 사용
            layer_spec = [n_layers] if isinstance(n_layers, int) else n_layers
            
            feats = model.get_intermediate_layers(
                image_resized_norm.unsqueeze(0).to(device), 
                n=layer_spec, 
                reshape=True, 
                norm=True
            )
            
            # 마지막 레이어 특징 사용
            x = feats[-1].squeeze().detach().cpu()
            
            # Spatial dimensions 저장 (C, H, W) -> (H, W)
            # ResNet의 경우 보통 (2048, 7, 7) 형태임
            if len(x.shape) == 3:
                h_patches, w_patches = x.shape[1], x.shape[2]
            else:
                # 1D feature인 경우 (Global Average Pooling 이후 등)
                h_patches, w_patches = 1, 1
            
            dim = x.shape[0]
            
            # 기존 코드와 동일한 reshape
            x = x.view(dim, -1).permute(1, 0)  # feature cnt, dims
    
    return x, h_patches, w_patches


def visualize_features_pca(features: torch.Tensor, h_patches: int, w_patches: int, original_image: Optional[torch.Tensor] = None, save_path: Optional[str] = None, n_components: int = 3):
    """
    PCA를 사용하여 특징을 시각화
    
    Args:
        features (torch.Tensor): 추출된 특징 (feature_cnt, dims)
        h_patches (int): 높이 패치 수
        w_patches (int): 너비 패치 수
        original_image (Optional[torch.Tensor]): 원본 이미지 (C, H, W)
        save_path (Optional[str]): 저장 경로
        n_components (int): PCA 차원 수 (1, 2, 3)
    """
    # PCA 적용
    pca = PCA(n_components=n_components, whiten=True)
    x_np = features.numpy()
    pca.fit(x_np)
    
    # PCA 변환
    projected_features = pca.transform(x_np)
    
    # 차원에 따른 시각화 처리
    if n_components == 1:
        # 1차원: Min-Max 정규화 후 Jet Colormap 적용
        proj_min, proj_max = projected_features.min(), projected_features.max()
        norm_features = (projected_features - proj_min) / (proj_max - proj_min + 1e-6)
        
        # (H, W) 형태로 변환
        heatmap = norm_features.reshape(h_patches, w_patches)
        
        # Colormap 적용 (numpy array 반환됨: H, W, 4)
        cmap = plt.get_cmap('jet')
        colored_map = cmap(heatmap)
        
        # RGB만 사용 (H, W, 3) -> (3, H, W)
        projected_image = torch.from_numpy(colored_map[:, :, :3]).permute(2, 0, 1).float()
        
    elif n_components == 2:
        # 2차원: R, G 채널에 매핑, B는 0
        projected_features_tensor = torch.from_numpy(projected_features).view(h_patches, w_patches, 2)
        
        # Sigmoid로 색상 강화 (기존 로직 활용)
        projected_features_tensor = torch.nn.functional.sigmoid(projected_features_tensor.mul(2.0))
        
        # Blue 채널 (0) 추가
        zeros = torch.zeros(h_patches, w_patches, 1)
        projected_image = torch.cat([projected_features_tensor, zeros], dim=2).permute(2, 0, 1)
        
    else: # n_components == 3
        # 3차원: 기존 로직 (RGB)
        projected_image = torch.from_numpy(projected_features).view(h_patches, w_patches, 3)
        projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    
    # 시각화
    plt.figure(figsize=(20, 8), dpi=300)
    
    # 원본 이미지 표시 (있을 경우)
    if original_image is not None:
        plt.subplot(1, 2, 1)
        # Tensor (C, H, W) -> Numpy (H, W, C)
        img_np = original_image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
    
    # PCA 결과 이미지 표시
    # Upsample for better visualization if patches are small
    if h_patches < 224:
        projected_image_up = torch.nn.functional.interpolate(
            projected_image.unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    else:
        projected_image_up = projected_image
        
    plt.imshow(projected_image_up.permute(1, 2, 0).numpy())
    plt.title(f"PCA Visualization (dims: {features.shape[1]} -> {n_components})")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"결과 저장됨: {save_path}")
    
    # plt.show() # 서버 환경에서는 주석 처리
    plt.close() # 메모리 누수 방지


# 비디오에서 특징 추출하는 함수
def extract_features_from_video(
    video_path: str,
    model_name: str = 'resnet50',
    n_layers: Union[int, List[int]] = 4,
    mask_path: Optional[str] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    roi_format: str = 'xywh',
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    device: str = 'cuda',
    output_dir: Optional[str] = None,
    pca_dim: int = 3
) -> List[torch.Tensor]:
    """
    비디오에서 프레임별로 특징 추출 (4가지 전처리 경우 지원)
    
    Args:
        video_path (str): 비디오 파일 경로
        model_name (str): 사용할 모델 이름
        n_layers (Union[int, List[int]]): 추출할 레이어 수 또는 특정 레이어 인덱스 리스트
        mask_path (Optional[str]): 마스크 파일 경로 (선택사항)
        roi (Optional[Tuple[int, int, int, int]]): ROI 좌표 (선택사항)
        roi_format (str): ROI 형식 ('xywh' 또는 'xyxy')
        start_frame (int): 시작 프레임 번호
        end_frame (Optional[int]): 종료 프레임 번호
        device (str): 디바이스
        output_dir (Optional[str]): 특징을 저장할 디렉토리 (선택사항)
    
    Returns:
        List[torch.Tensor]: 프레임별 추출된 특징 리스트
    """
    import sys
    import os
    from pathlib import Path
    
    # utils 모듈 import
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.read_video import VideoFrameReader, VideoMaskProcessor, VideoROIProcessor, VideoMaskROIProcessor
    
    # 출력 디렉토리 생성: video_name/model_name/layer_number
    if output_dir:
        # 비디오 이름 추출 (확장자 제외)
        video_name = Path(video_path).stem
        
        # 레이어 번호 결정
        if isinstance(n_layers, int):
            # n_layers가 int면 마지막 레이어 번호 사용
            layer_number = n_layers - 1
        elif isinstance(n_layers, list):
            # list면 마지막 레이어 인덱스 사용
            layer_number = n_layers[-1]
        else:
            layer_number = 0
        
        # 최종 출력 경로: output_dir/video_name/model_name/mode/layer_number/pca{dim}
        if mask_path and roi:
            mode_dir = "mask_roi"
        elif mask_path:
            mode_dir = "mask"
        elif roi:
            mode_dir = "roi"
        else:
            mode_dir = "normal" # 또는 빈 문자열, 하지만 구조 통일성을 위해 명시
            
        # normal 모드일 경우 경로를 줄일지 여부는 선택사항이지만, 
        # 사용자 요청 예시(.../roi/1/pca3)에 맞추려면 mode 디렉토리가 있는 것이 좋음
        if mode_dir == "normal":
             final_output_dir = Path(output_dir) / video_name / model_name / str(layer_number) / f"pca{pca_dim}"
        else:
             final_output_dir = Path(output_dir) / video_name / model_name / mode_dir / str(layer_number) / f"pca{pca_dim}"
             
        final_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 출력 디렉토리: {final_output_dir}")
    else:
        final_output_dir = None
    
    # 모델 생성
    model = create_model(model_name, pretrained=True, device=device)
    
    # 이미지 전처리 변환
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_features = []
    
    # 4가지 경우에 따라 처리
    # Case 1: 마스크 + ROI
    if mask_path and roi:
        print(f"🎬 비디오 + 마스크 + ROI 처리: {video_path}")
        print(f"   마스크: {mask_path}")
        print(f"   ROI: {roi} (형식: {roi_format})")
        
        with VideoMaskROIProcessor(video_path, mask_path, roi, roi_format) as processor:
            video_info = processor.get_video_info()
            print(f"비디오 정보: {video_info['frame_count']}프레임, {video_info['width']}x{video_info['height']}")
            print(f"ROI 정보: {video_info['roi']}")
            
            for frame_num, original_frame, processed_frame in processor.process_frames_with_mask_and_roi(start_frame, end_frame):
                # 전처리된 프레임(마스크+ROI) 사용
                frame_tensor = transform(processed_frame)
                
                # 특징 추출 (모델 재사용)
                features, h, w = extract_features_like_original(frame_tensor, model_name, n_layers, device, model=model)
                all_features.append(features)
                
                print(f"  프레임 {frame_num}: 특징 shape {features.shape}, Grid: {h}x{w}")
                
                # 시각화 및 저장 (전처리된 이미지 사용)
                if final_output_dir:
                    vis_path = final_output_dir / f"frame_{frame_num:04d}_pca.png"
                    visualize_features_pca(features, h, w, frame_tensor, str(vis_path), n_components=pca_dim)
    
    # Case 2: 마스크만
    elif mask_path:
        print(f"🎬 비디오 + 마스크 처리: {video_path}")
        print(f"   마스크: {mask_path}")
        
        with VideoMaskProcessor(video_path, mask_path) as processor:
            video_info = processor.get_video_info()
            print(f"비디오 정보: {video_info['frame_count']}프레임, {video_info['width']}x{video_info['height']}")
            
            for frame_num, original_frame, processed_frame in processor.process_frames_with_mask(start_frame, end_frame):
                # 전처리된 프레임(마스크) 사용
                frame_tensor = transform(processed_frame)
                
                # 특징 추출 (모델 재사용)
                features, h, w = extract_features_like_original(frame_tensor, model_name, n_layers, device, model=model)
                all_features.append(features)
                
                print(f"  프레임 {frame_num}: 특징 shape {features.shape}, Grid: {h}x{w}")
                
                # 시각화 및 저장 (전처리된 이미지 사용)
                if final_output_dir:
                    vis_path = final_output_dir / f"frame_{frame_num:04d}_pca.png"
                    visualize_features_pca(features, h, w, frame_tensor, str(vis_path), n_components=pca_dim)
    
    # Case 3: ROI만
    elif roi:
        print(f"🎬 비디오 + ROI 처리: {video_path}")
        print(f"   ROI: {roi} (형식: {roi_format})")
        
        with VideoROIProcessor(video_path, roi, roi_format) as processor:
            video_info = processor.get_video_info()
            print(f"비디오 정보: {video_info['frame_count']}프레임, {video_info['width']}x{video_info['height']}")
            print(f"ROI 정보: {video_info['roi']}")
            
            for frame_num, original_frame, processed_frame in processor.process_frames_with_roi(start_frame, end_frame):
                # 전처리된 프레임(ROI) 사용
                frame_tensor = transform(processed_frame)
                
                # 특징 추출 (모델 재사용)
                features, h, w = extract_features_like_original(frame_tensor, model_name, n_layers, device, model=model)
                all_features.append(features)
                
                print(f"  프레임 {frame_num}: 특징 shape {features.shape}, Grid: {h}x{w}")
                
                # 시각화 및 저장 (전처리된 이미지 사용)
                if final_output_dir:
                    vis_path = final_output_dir /f"frame_{frame_num:04d}_pca.png"
                    visualize_features_pca(features, h, w, frame_tensor, str(vis_path), n_components=pca_dim)
    
    # Case 4: 기본 (전처리 없음)
    else:
        print(f"🎬 비디오 처리 (전처리 없음): {video_path}")
        # read_video 제너레이터 사용
        from utils.read_video import read_video
        
        # 비디오 정보는 cv2로 직접 확인
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"비디오 정보: {frame_count}프레임, {width}x{height}")
        
        for frame_num, frame in enumerate(read_video(video_path)):
            # 범위 체크
            if frame_num < start_frame:
                continue
            if end_frame is not None and frame_num >= end_frame:
                break
                
            # 프레임 전처리
            frame_tensor = transform(frame)
            
            # 특징 추출 (모델 재사용)
            features, h, w = extract_features_like_original(frame_tensor, model_name, n_layers, device, model=model)
            all_features.append(features)
            
            print(f"  프레임 {frame_num}: 특징 shape {features.shape}, Grid: {h}x{w}")
            
            # 시각화 및 저장
            if final_output_dir:
                vis_path = final_output_dir  / f"frame_{frame_num:04d}_pca.png"
                visualize_features_pca(features, h, w, frame_tensor, str(vis_path))
    
    print(f"\n✅ 총 {len(all_features)}개 프레임 처리 완료")
    return all_features



# 다양한 모델에서 특징 추출하는 함수
def compare_model_features(image: torch.Tensor, models_to_test: List[str] = None) -> dict:
    """
    여러 모델에서 특징을 추출하고 비교
    
    Args:
        image (torch.Tensor): 입력 이미지 (C, H, W)
        models_to_test (List[str]): 테스트할 모델 리스트
    
    Returns:
        dict: 모델별 특징 정보
    """
    if models_to_test is None:
        models_to_test = ['resnet50', 'resnet101', 'mobilenet_v2', 'vgg16', 'vgg19']
    
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_name in models_to_test:
        try:
            print(f"Testing {model_name}...")
            features, h, w = extract_features_like_original(image, model_name, n_layers=4, device=device)
            
            results[model_name] = {
                'shape': features.shape,
                'feature_count': features.shape[0],
                'feature_dims': features.shape[1],
                'mean': features.mean().item(),
                'std': features.std().item()
            }
            
            print(f"  {model_name}: {features.shape} (feature_cnt, dims)")
            
        except Exception as e:
            print(f"  {model_name}: Error - {e}")
            results[model_name] = {'error': str(e)}
    
    return results


# 사용 예시 및 테스트
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='비디오에서 CNN 특징 추출')
    parser.add_argument('--video', type=str, default='/home/kjm/foreground_segmentation/dataset/video/all normal 2.avi', \
                                                                                            help='비디오 파일 경로')
    parser.add_argument('--mask', type=str, default=None, help='마스크 파일 경로 (선택사항)')
    parser.add_argument('--roi', type=int, nargs=4, default=None, metavar=('X', 'Y', 'W', 'H'),
                       help='ROI 좌표 (예: --roi 100 100 300 300)')
    parser.add_argument('--roi-format', type=str, default='xywh', choices=['xywh', 'xyxy'],
                       help='ROI 형식 (xywh: x,y,width,height 또는 xyxy: x1,y1,x2,y2)')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                               'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                               'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn'],
                       help='사용할 모델')
    parser.add_argument('--layers', type=int, default=4, help='추출할 레이어 수 (--layer-indices와 함께 사용 불가)')
    parser.add_argument('--layer-indices', type=int, nargs='+', default=None, 
                       help='추출할 특정 레이어 인덱스 (예: --layer-indices 2 3)')
    parser.add_argument('--start', type=int, default=0, help='시작 프레임')
    parser.add_argument('--end', type=int, default=None, help='종료 프레임')
    parser.add_argument('--output', type=str, default='vis_result', help='특징 저장 디렉토리')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='디바이스')
    parser.add_argument('--demo', action='store_true', help='데모 모드 (더미 이미지 테스트)')
    parser.add_argument('--compare', action='store_true', help='여러 모델 비교 모드')
    parser.add_argument('--pca-dim', type=int, default=2, choices=[1, 2, 3], help='PCA 축소 차원 (1, 2, 3)')
    
    args = parser.parse_args()
    
    # layer-indices와 layers 동시 사용 체크
    if args.layer_indices is not None and args.layers != 4:
        print("⚠️  --layers와 --layer-indices를 동시에 사용할 수 없습니다.")
        print("   --layer-indices를 사용합니다.")
    
    # 레이어 설정 결정
    layer_config = args.layer_indices if args.layer_indices is not None else args.layers
    
    # ROI를 튜플로 변환
    roi_tuple = tuple(args.roi) if args.roi else None
    
    print("=== 비디오에서 특징 추출 ===")
    if isinstance(layer_config, list):
        print(f"📌 추출할 레이어: {layer_config}")
    else:
        print(f"📌 추출할 레이어 수: {layer_config} (레이어 0-{layer_config-1})")
    
    # 전처리 모드 출력
    if args.mask and roi_tuple:
        print(f"📌 전처리 모드: 마스크 + ROI")
    elif args.mask:
        print(f"📌 전처리 모드: 마스크만")
    elif roi_tuple:
        print(f"📌 전처리 모드: ROI만")
    else:
        print(f"📌 전처리 모드: 없음 (원본 프레임)")
    print()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 실행합니다.")
    
    try:
        features = extract_features_from_video(
            video_path=args.video,
            model_name=args.model,
            n_layers=layer_config,
            mask_path=args.mask,
            roi=roi_tuple,
            roi_format=args.roi_format,
            start_frame=args.start,
            end_frame=args.end,
            device=device,
            output_dir=args.output,
            pca_dim=args.pca_dim
        )
        
        print(f"\n📊 추출 결과:")
        print(f"  총 프레임 수: {len(features)}")
        print(f"  특징 shape: {features[0].shape if features else 'N/A'}")
        
        if args.output:
            print(f"  저장 위치: {args.output}/")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
