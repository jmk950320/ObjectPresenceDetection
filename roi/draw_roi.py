import cv2
import sys
import os
import argparse
from pathlib import Path
import yaml

# 상위 디렉토리를 path에 추가하여 utils 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.read_video import VideoFrameReader
except ImportError:
    print("Error: utils.read_video를 찾을 수 없습니다. 경로 설정을 확인하세요.")
    sys.exit(1)

def select_roi(video_path: str, output_path: str = "roi_result.yaml"):
    """
    비디오의 첫 프레임에서 ROI를 선택하고 저장합니다.
    """
    try:
        # VideoFrameReader를 사용하여 첫 프레임 읽기
        with VideoFrameReader(video_path) as reader:
            info = reader.get_video_info()
            print(f"\n=== Video Information ===")
            print(f"Path: {video_path}")
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"Total Frames: {info['frame_count']}")
            print(f"FPS: {info['fps']}")
            print("=" * 30)

            # 첫 번째 프레임 가져오기 (frame_number=0)
            # VideoFrameReader.read_frame_at은 RGB를 반환하므로 cv2 디스플레이를 위해 BGR로 변환 필요
            first_frame_rgb = reader.read_frame_at(0)
            if first_frame_rgb is None:
                print("Error: 첫 번째 프레임을 읽을 수 없습니다.")
                return

            first_frame_bgr = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)

        # ROI 선택 창 생성
        window_name = "Select ROI - Press ENTER or SPACE to confirm, c to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # roi = (x, y, w, h)
        roi = cv2.selectROI(window_name, first_frame_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)

        if roi == (0, 0, 0, 0):
            print("ROI 선택이 취소되었습니다.")
            return

        x, y, w, h = roi
        x1, y1, x2, y2 = x, y, x + w, y + h

        print("\n=== Selected ROI ===")
        print(f"Format [x, y, w, h] (xywh): [{x}, {y}, {w}, {h}]")
        print(f"Format [x1, y1, x2, y2] (xyxy): [{x1}, {y1}, {x2}, {y2}]")
        print("=" * 20)

        # 결과 저장 (YAML 형식)
        result = {
            'video_path': str(video_path),
            'roi_xywh': [int(x), int(y), int(w), int(h)],
            'roi_xyxy': [int(x1), int(y1), int(x2), int(y2)],
            'resolution': [info['width'], info['height']]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        print(f"결과가 {output_path}에 저장되었습니다.")

        # test_config.yaml에 적용할 수 있는 형식 출력
        print("\n--- Copy this to your config file ---")
        print(f"roi: [{x1}, {y1}, {x2}, {y2}]")
        print("roi_format: xyxy")
        print("--------------------------------------")

    except Exception as e:
        print(f"오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description='비디오 첫 프레임에서 ROI를 선택하는 도구')
    parser.add_argument('--video',type=str, default="/home/kjm/foreground_segmentation/dataset/video/test4.avi", help='비디오 파일 경로')


    args = parser.parse_args()

    video_path = args.video
    output_path = f"{Path(video_path).stem}_roi.yaml"

    select_roi(video_path, output_path)

if __name__ == "__main__":
    main()
