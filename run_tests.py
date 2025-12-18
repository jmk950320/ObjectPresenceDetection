#!/usr/bin/env python3
"""
비디오 읽기 유틸리티 테스트 실행 스크립트

사용법:
    python run_tests.py                    # 모든 테스트 실행
    python run_tests.py --class VideoFrameReader  # 특정 클래스만 테스트
    python run_tests.py --verbose          # 상세 출력
"""

import sys
import os
import argparse
import unittest
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_specific_test_class(class_name, verbose=False):
    """특정 테스트 클래스만 실행"""
    try:
        from test.test_video_reading import (
            TestVideoFrameReader, 
            TestVideoMaskProcessor, 
            TestProcessVideoWithMaskExample, 
            TestIntegration
        )
        
        class_map = {
            'VideoFrameReader': TestVideoFrameReader,
            'VideoMaskProcessor': TestVideoMaskProcessor,
            'ProcessVideoWithMaskExample': TestProcessVideoWithMaskExample,
            'Integration': TestIntegration
        }
        
        if class_name not in class_map:
            print(f"❌ 테스트 클래스 '{class_name}'를 찾을 수 없습니다.")
            print(f"사용 가능한 클래스: {', '.join(class_map.keys())}")
            return False
        
        test_class = class_map[class_name]
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        verbosity = 2 if verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ 테스트 모듈을 가져올 수 없습니다: {e}")
        return False

def run_all_tests(verbose=False):
    """모든 테스트 실행"""
    try:
        from test.test_video_reading import run_tests
        return run_tests()
    except ImportError as e:
        print(f"❌ 테스트 모듈을 가져올 수 없습니다: {e}")
        return False

def check_dependencies():
    """필요한 의존성 확인"""
    required_packages = ['cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print("pip install opencv-python numpy")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='비디오 읽기 유틸리티 테스트 실행')
    parser.add_argument('--class', dest='test_class', 
                       help='실행할 특정 테스트 클래스 (VideoFrameReader, VideoMaskProcessor, ProcessVideoWithMaskExample, Integration)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 출력 모드')
    parser.add_argument('--check-deps', action='store_true',
                       help='의존성만 확인하고 종료')
    
    args = parser.parse_args()
    
    # 의존성 확인
    if not check_dependencies():
        return 1
    
    if args.check_deps:
        print("✅ 모든 의존성이 설치되어 있습니다.")
        return 0
    
    print("=== 비디오 읽기 유틸리티 테스트 ===")
    
    # 테스트 실행
    if args.test_class:
        print(f"특정 클래스 테스트: {args.test_class}")
        success = run_specific_test_class(args.test_class, args.verbose)
    else:
        print("모든 테스트 실행")
        success = run_all_tests(args.verbose)
    
    if success:
        print("\n✅ 테스트가 성공적으로 완료되었습니다!")
        return 0
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
        return 1

if __name__ == "__main__":
    sys.exit(main())