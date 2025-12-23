import cv2
import yaml
import argparse
import os
import sys
from pathlib import Path

def confirm_roi(config_path: str, video_override: str = None):
    """
    YAML 파일에서 ROI 정보를 읽어와 비디오에 오버레이하여 표시합니다.
    """
    if not os.path.exists(config_path):
        print(f"Error: 설정 파일을 찾을 수 없습니다: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    video_path = video_override if video_override else config.get('video_path')
    if not video_path:
        print("Error: 비디오 경로를 찾을 수 없습니다. YAML의 'video_path' 또는 --video 인자를 확인하세요.")
        return

    if not os.path.exists(video_path):
        print(f"Error: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return

    # ROI 정보 추출
    # roi_xyxy를 우선적으로 사용, 없으면 roi_xywh 사용
    roi_xyxy = config.get('roi_xyxy')
    roi_xywh = config.get('roi_xywh')

    if roi_xyxy:
        x1, y1, x2, y2 = roi_xyxy
    elif roi_xywh:
        x, y, w, h = roi_xywh
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        print("Error: YAML 파일에 'roi_xyxy' 또는 'roi_xywh' 정보가 없습니다.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 비디오를 열 수 없습니다: {video_path}")
        return

    print(f"\n=== ROI Confirmation ===")
    print(f"Video: {video_path}")
    print(f"ROI (xyxy): [{x1}, {y1}, {x2}, {y2}]")
    print("Press 'q' or 'ESC' to exit.")
    print("=" * 25)

    window_name = f"Confirm ROI - {Path(video_path).name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            # 비디오 끝에 도달하면 처음으로 되돌림 (반복 재생)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ROI 사각형 그리기 (빨간색, 두께 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 좌표 텍스트 추가
        text = f"ROI: [{x1}, {y1}, {x2}, {y2}]"
        cv2.putText(frame, text, (x1, max(y1 - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YAML 설정 파일의 ROI를 비디오에서 확인합니다.')
    parser.add_argument('--config', type=str, required=True, help='ROI 정보가 담긴 YAML 파일 경로')
    parser.add_argument('--video', type=str, help='(선택사항) 비디오 파일 경로 override')

    args = parser.parse_args()
    confirm_roi(args.config, args.video)

if __name__ == "__main__":
    main()
