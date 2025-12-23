"""
ROI Selection Tool for Images and Videos using cv2.setMouseCallback
"""
import cv2
import numpy as np
import argparse
from pathlib import Path


class ROISelector:
    def __init__(self, source_path):
        """
        Initialize ROI Selector
        
        Args:
            source_path: Path to image or video file
        """
        self.source_path = source_path
        self.is_video = self._check_if_video(source_path)
        
        # ROI selection state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_list = []
        self.current_frame = None
        self.display_frame = None
        
        # Window name
        self.window_name = "ROI Selection - Press 'r' to reset, 's' to save, 'q' to quit"
    
    def _check_if_video(self, path):
        """Check if the source is a video file"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        return Path(path).suffix.lower() in video_extensions
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for ROI selection
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing rectangle
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update rectangle while dragging
            if self.drawing:
                self.end_point = (x, y)
                self._update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing rectangle
            self.drawing = False
            self.end_point = (x, y)
            
            # Calculate ROI coordinates (ensure top-left and bottom-right)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            # Only add if rectangle has area
            if x2 - x1 > 5 and y2 - y1 > 5:
                roi = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                self.roi_list.append(roi)
                print(f"ROI #{len(self.roi_list)} added: ({x1}, {y1}) -> ({x2}, {y2})")
            
            self._update_display()
    
    def _update_display(self):
        """Update the display with current ROI rectangles"""
        self.display_frame = self.current_frame.copy()
        
        # Draw all saved ROIs in green
        for idx, roi in enumerate(self.roi_list):
            cv2.rectangle(
                self.display_frame,
                (roi['x1'], roi['y1']),
                (roi['x2'], roi['y2']),
                (0, 255, 0),
                2
            )
            # Add ROI number label
            cv2.putText(
                self.display_frame,
                f"ROI #{idx + 1}",
                (roi['x1'], roi['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw current rectangle being drawn (in red)
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(
                self.display_frame,
                self.start_point,
                self.end_point,
                (0, 0, 255),
                2
            )
        
        cv2.imshow(self.window_name, self.display_frame)
    
    def reset_roi(self):
        """Reset all ROI selections"""
        self.roi_list = []
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self._update_display()
        print("All ROIs reset")
    
    def save_roi(self, output_path="roi_data.txt"):
        """
        Save ROI data to file
        
        Args:
            output_path: Path to save ROI data
        """
        if not self.roi_list:
            print("No ROI to save!")
            return
        
        with open(output_path, 'w') as f:
            f.write(f"Source: {self.source_path}\n")
            f.write(f"Total ROIs: {len(self.roi_list)}\n")
            f.write("-" * 50 + "\n")
            
            for idx, roi in enumerate(self.roi_list):
                f.write(f"ROI #{idx + 1}:\n")
                f.write(f"  Top-Left: ({roi['x1']}, {roi['y1']})\n")
                f.write(f"  Bottom-Right: ({roi['x2']}, {roi['y2']})\n")
                f.write(f"  Width: {roi['width']}, Height: {roi['height']}\n")
                f.write("-" * 50 + "\n")
        
        print(f"ROI data saved to {output_path}")
        
        # Also save as numpy array for easy loading
        roi_array = np.array([[roi['x1'], roi['y1'], roi['x2'], roi['y2']] 
                              for roi in self.roi_list])
        np.save(output_path.replace('.txt', '.npy'), roi_array)
        print(f"ROI array saved to {output_path.replace('.txt', '.npy')}")
    
    def run_image(self):
        """Run ROI selection on image"""
        # Read image
        self.current_frame = cv2.imread(self.source_path)
        if self.current_frame is None:
            print(f"Error: Cannot read image from {self.source_path}")
            return
        
        self.display_frame = self.current_frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== ROI Selection Mode ===")
        print("Instructions:")
        print("  - Click and drag to draw ROI rectangle")
        print("  - Press 'r' to reset all ROIs")
        print("  - Press 's' to save ROIs")
        print("  - Press 'q' to quit")
        print("=" * 30 + "\n")
        
        cv2.imshow(self.window_name, self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                self.reset_roi()
            elif key == ord('s'):
                self.save_roi()
        
        cv2.destroyAllWindows()
    
    def run_video(self):
        """Run ROI selection on video (first frame)"""
        cap = cv2.VideoCapture(self.source_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video from {self.source_path}")
            return
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read first frame from video")
            cap.release()
            return
        
        self.current_frame = frame
        self.display_frame = frame.copy()
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n=== Video Information ===")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print("=" * 30 + "\n")
        
        cap.release()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("=== ROI Selection Mode (First Frame) ===")
        print("Instructions:")
        print("  - Click and drag to draw ROI rectangle")
        print("  - Press 'r' to reset all ROIs")
        print("  - Press 's' to save ROIs")
        print("  - Press 'q' to quit")
        print("=" * 30 + "\n")
        
        cv2.imshow(self.window_name, self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                self.reset_roi()
            elif key == ord('s'):
                self.save_roi()
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Run ROI selection based on source type"""
        if self.is_video:
            self.run_video()
        else:
            self.run_image()
        
        # Print final ROI summary
        if self.roi_list:
            print(f"\n=== Final ROI Summary ===")
            print(f"Total ROIs selected: {len(self.roi_list)}")
            for idx, roi in enumerate(self.roi_list):
                print(f"ROI #{idx + 1}: [{roi['x1']}, {roi['y1']}, {roi['x2']}, {roi['y2']}]")
            print("=" * 30 + "\n")


def main():
    parser = argparse.ArgumentParser(description='ROI Selection Tool for Images and Videos')
    parser.add_argument('source', type=str, help='Path to image or video file')
    parser.add_argument('-o', '--output', type=str, default='roi_data.txt',
                        help='Output file path for ROI data (default: roi_data.txt)')
    
    args = parser.parse_args()
    
    # Check if source file exists
    if not Path(args.source).exists():
        print(f"Error: Source file '{args.source}' does not exist!")
        return
    
    # Create ROI selector and run
    selector = ROISelector(args.source)
    selector.run()


if __name__ == "__main__":
    main()
