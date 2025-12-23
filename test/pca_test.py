import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
import yaml
# feature_visualizetion 디렉토리를 path에 추가하여 모듈 임포트 가능하게 함
sys.path.insert(0, str(Path(__file__).parent.parent / "feature_visualizetion"))
from vis_resnet import create_model, extract_features_like_original

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_roi_coords(roi, roi_format, h_img, w_img):
    """
    ROI를 (x1, y1, x2, y2) 정수 좌표로 변환하여 반환
    """
    if not roi:
        return 0, 0, w_img, h_img
    if roi_format == 'xyxy':
        x1, y1, x2, y2 = roi
    elif roi_format == 'xywh':
        x1, y1, w_roi, h_roi = roi
        x2, y2 = x1 + w_roi, y1 + h_roi
    else:
        return 0, 0, w_img, h_img

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
    return x1, y1, x2, y2

def apply_mask_and_roi(grid_h, grid_w, mask_path, roi, roi_format, original_size=None):
    """
    Apply mask and ROI to the feature map grid.
    Returns a boolean mask of shape (grid_h, grid_w)
    """
    H, W = grid_h, grid_w
    valid_mask = np.ones((H, W), dtype=bool)
    
    if (mask_path and os.path.exists(mask_path)) or roi:
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # If no mask image but ROI exists, create a white mask if original_size is known
            if roi and original_size:
                oh, ow = original_size
                mask_img = np.full((oh, ow), 255, dtype=np.uint8)
            else:
                mask_img = None

        if mask_img is not None:
            if roi:
                h_img, w_img = mask_img.shape[:2]
                x1, y1, x2, y2 = get_roi_coords(roi, roi_format, h_img, w_img)
                mask_img = mask_img[y1:y2, x1:x2]

            if mask_img.size > 0:
                mask_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
                valid_mask = valid_mask & (mask_resized > 127)
            else:
                valid_mask[:] = False
            
    return valid_mask

def load_pca_params(pca_path):
    """
    PCA components와 mean을 로드함
    """
    data = np.load(pca_path)
    components = data['pca_components'] # (n_components, dim)
    mean = data['pca_mean']             # (dim,)
    print(f"✅ PCA 파라미터 로드됨: components {components.shape}, mean {mean.shape}")
    return components, mean

def apply_pca_manual(features, components, mean):
    """
    수동으로 PCA 변환 수행: (features - mean) @ components.T
    features: (N, dim) numpy array 또는 torch.Tensor
    """
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    
    # 정규화 (mean subtraction)
    centered_features = features - mean
    
    # 투영 (dot product with components)
    projected = np.dot(centered_features, components.T)
    
    return projected

def visualize_projected_features(projected, h, w, original_image=None, save_path=None, valid_mask=None):
    """
    PCA 투영된 특징을 시각화 (RGB 매핑)
    projected: (N, 3) 
    """
    n_components = projected.shape[1]
    
    # 시각화를 위해 [0, 1] 범위를 맞춤 (Min-Max 또는 Sigmoid)
    # If valid_mask is provided, calculate min/max only from valid regions
    if valid_mask is not None:
        flat_mask = valid_mask.flatten()
        if np.any(flat_mask):
            p_min = projected[flat_mask].min(axis=0)
            p_max = projected[flat_mask].max(axis=0)
        else:
            p_min = projected.min(axis=0)
            p_max = projected.max(axis=0)
    else:
        p_min = projected.min(axis=0)
        p_max = projected.max(axis=0)

    normalized = (projected - p_min) / (p_max - p_min + 1e-8)
    
    # (H, W, C) 형태로 변환
    if n_components >= 3:
        vis_img = normalized[:, :3].reshape(h, w, 3)
    elif n_components == 2:
        vis_img = np.zeros((h, w, 3))
        vis_img[:, :, :2] = normalized.reshape(h, w, 2)
    else: # 1
        vis_img = normalized.reshape(h, w)
        cmap = plt.get_cmap('jet')
        vis_img = cmap(vis_img)[:, :, :3]
    
    # Apply valid mask
    if valid_mask is not None:
        vis_img = vis_img * valid_mask[:, :, np.newaxis]

    # Apply interpolation
    vis_img_up = cv2.resize(vis_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    plt.figure(figsize=(10, 5))
    
    if original_image is not None:
        plt.subplot(1, 2, 1)
        if torch.is_tensor(original_image):
            img_np = original_image.permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            plt.imshow(img_np)
        else:
            plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
    
    # Upsample for better view
    vis_img_up = cv2.resize(vis_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    plt.imshow(vis_img_up)
    plt.title(f"Manual PCA Visualization ({n_components} components)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ 시각화 결과 저장됨: {save_path}")
    
    plt.show()
    plt.close()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def render_pca_visualization(projected, h, w, original_image_tensor=None, valid_mask=None):
    """
    PCA 투영된 특징을 시각화하여 numpy array(BGR)로 반환 (비디오 프레임용)
    """
    n_components = projected.shape[1]
        
    normalized = sigmoid(projected * 2)
    
    # (H, W, C) array creation
    if n_components >= 3:
        vis_img = normalized[:, :3].reshape(h, w, 3)
    elif n_components == 2:
        vis_img = np.zeros((h, w, 3), dtype=np.float32)
        vis_img[:, :, :2] = normalized.reshape(h, w, 2)
    else: # 1
        vis_img = normalized.reshape(h, w)
        cmap = plt.get_cmap('jet')
        vis_img = cmap(vis_img)[:, :, :3]
    
    # Apply valid mask (zero out invalid regions)
    if valid_mask is not None:
        vis_img = vis_img * valid_mask[:, :, np.newaxis]

    # Resize PCA visualization to 224x224
    vis_img_up = cv2.resize(vis_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    vis_img_bgr = cv2.cvtColor((vis_img_up * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    if original_image_tensor is not None:
        # Denormalize and convert original_image_tensor to BGR
        img_np = original_image_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Concatenate side-by-side
        combined = np.hstack((img_bgr, vis_img_bgr))
        return combined
    
    return vis_img_bgr

def process_video(video_path, model, device, components, pca_mean, args, output_path=None):
    """
    비디오 파일에 대해 프레임별 특징 추출, PCA 변환, 시각화 수행 및 비디오 저장
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output size is (448, 224) because we stack (224, 224) horizontal
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (448, 224))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"🎥 비디오 처리 시작: {video_path} ({total_frames} frames)")
    
    # ROI 적용 좌표 계산
    x1, y1, x2, y2 = get_roi_coords(args.roi, args.roi_format, orig_h, orig_w)
    
    # Pre-calculate valid_mask
    ret, first_frame = cap.read()
    if not ret:
        print("❌ 비디오 프레임을 읽을 수 없습니다.")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # ROI Crop
    first_frame_cropped = first_frame[y1:y2, x1:x2]
    img_rgb = cv2.cvtColor(first_frame_cropped, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb)
    with torch.inference_mode():
        _, h_grid, w_grid = extract_features_like_original(
            img_tensor, 
            model_name=args.model, 
            n_layers=[args.layer], 
            device=device, 
            model=model
        )
    valid_mask = apply_mask_and_roi(h_grid, w_grid, args.mask, args.roi, args.roi_format, original_size=(orig_h, orig_w))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # ROI Crop
        frame_cropped = frame[y1:y2, x1:x2]
        img_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb)
        
        # 특징 추출
        with torch.inference_mode():
            features, h, w = extract_features_like_original(
                img_tensor, 
                model_name=args.model, 
                n_layers=[args.layer], 
                device=device, 
                model=model
            )
        
        # PCA 변환
        projected = apply_pca_manual(features, components, pca_mean)
        
        # 시각화 렌더링 (valid_mask 적용)
        combined_frame = render_pca_visualization(projected, h, w, img_tensor, valid_mask=valid_mask)
        
        if out:
            out.write(combined_frame)
            
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    if out:
        out.release()
    print(f"✅ 비디오 처리 완료: {output_path}")

def process_single_image(image_path, model, device, components, pca_mean, args, output_path=None):
    """
    이미지 하나에 대해 특징 추출, PCA 변환, 시각화 수행
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return
    # ROI 적용 좌표 계산
    x1, y1, x2, y2 = get_roi_coords(args.roi, args.roi_format, orig_h, orig_w)
    
    # ROI Crop
    img_cropped = img[y1:y2, x1:x2]
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb)
    
    # 특징 추출
    features, h, w = extract_features_like_original(
        img_tensor, 
        model_name=args.model, 
        n_layers=[args.layer], 
        device=device, 
        model=model
    )
    
    # PCA 변환
    projected = apply_pca_manual(features, components, pca_mean)
    
    # ROI 및 Mask 적용
    valid_mask = apply_mask_and_roi(h, w, args.mask, args.roi, args.roi_format, original_size=(orig_h, orig_w))
    
    # 시각화 및 저장
    # visualize_projected_features는 matplotlib 기반이라 valid_mask 적용을 위해 수정이 필요하거나
    # projected를 미리 필터링해야 함. 여기서는 projected를 필터링하는 방식으로 대응.
    
    # projected: (N, C)
    if valid_mask is not None:
        flat_mask = valid_mask.flatten()
        # 시각화를 위해 마스크 밖은 0 (또는 min값)으로 처리
        # visualize_projected_features 내에서 min-max를 하므로, 
        # 마스크 밖을 p_min으로 채우면 시각화에서 검게 나옴.
        # 하지만 visualize_projected_features 내부 로직을 바꾸는게 깔끔함.
        pass

    # 여기서는 간단하게 render_pca_visualization을 사용하여 저장할 수도 있음.
    # 만약 matplotlib 기반 visualize_projected_features를 유지하고 싶다면 해당 함수도 수정 필요.
    # 일단은 render_pca_visualization 결과를 저장하는 방식으로 교체하거나 수정 제안.
    
    # visualize_projected_features 내부에서 valid_mask를 처리하도록 수정하는 것이 좋겠음.
    visualize_projected_features(projected, h, w, img_tensor, output_path, valid_mask=valid_mask)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PCA 파라미터를 이용한 특징 추출 및 시각화 테스트')
    parser.add_argument('--config', type=str, help='설정 파일 경로 (YAML)')
    args = parser.parse_args()

    # 설정 파일 로드
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        print(f"📄 설정 파일 로드됨: {args.config}")

    # 인자 병합 (CLI가 우선)
    args.image = config.get('image') or config.get('video') # video 키도 지원
    args.pca = config.get('pca', '/home/kjm/foreground_segmentation/sampling_result/aggregated_sampled_pca.npz')
    args.model = config.get('model', 'resnet50')
    args.layer = config.get('layer_indices', config.get('layer', 2)) # Support both names
    args.roi = config.get('roi')
    args.roi_format =  config.get('roi_format', 'xyxy')
    args.mask = config.get('mask', None)
    args.output = config.get('output', 'pca_test_result.mp4' if 'video' in config or (args.image and Path(args.image).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')) else 'pca_test_result.png')

    if not args.image:
        print("❌ 입력 이미지(또는 폴더) 경로가 필요합니다.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. PCA 파라미터 로드
    if not os.path.exists(args.pca):
        print(f"❌ PCA 파일을 찾을 수 없습니다: {args.pca}")
        return
    components, pca_mean = load_pca_params(args.pca)
    
    # 2. 모델 로드
    print(f"🚀 모델({args.model}) 로드 중...")
    model = create_model(args.model, pretrained=True, device=device)
    
    # 3. 입력 처리 (파일 vs 폴더)
    input_path = Path(args.image)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    if input_path.is_dir():
        # 폴더 내 이미지 검색
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_extensions])
        
        if not image_files:
            print(f"❌ 폴더 내 이미지가 없습니다: {args.image}")
            return
            
        # 출력 폴더 생성
        output_dir = Path(args.output)
        if output_dir.suffix: # 파일 경로인 경우 부모 디렉토리에 폴더 생성
            output_dir = output_dir.parent / "pca_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 폴더 처리 시작: {len(image_files)}개 이미지, 저장: {output_dir}")
        
        for i, img_path in enumerate(image_files):
            out_path = output_dir / f"{img_path.stem}_pca.png"
            process_single_image(img_path, model, device, components, pca_mean, args, out_path)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")
    elif input_path.suffix.lower() in video_extensions:
        # 비디오 처리
        process_video(input_path, model, device, components, pca_mean, args, args.output)
    else:
        # 단일 이미지 처리
        process_single_image(input_path, model, device, components, pca_mean, args, args.output)

if __name__ == "__main__":
    main()
