import cv2
import numpy as np
import argparse
import os

# Global variables
drawing = False
ix, iy = -1, -1
img = None
clean_img = None # Original image without any drawings
ellipse_params = None # Dictionary to store 'center', 'axes', 'angle'

def draw_ellipse(event, x, y, flags, param):
    global ix, iy, drawing, ellipse_params

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Calculate center and axes dynamically while dragging
            center = ((ix + x) // 2, (iy + y) // 2)
            axes = (abs(x - ix) // 2, abs(y - iy) // 2)
            ellipse_params = {'center': center, 'axes': axes, 'angle': 0}

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Finalize the ellipse parameters
        center = ((ix + x) // 2, (iy + y) // 2)
        axes = (abs(x - ix) // 2, abs(y - iy) // 2)
        ellipse_params = {'center': center, 'axes': axes, 'angle': 0}

def main():
    global img, clean_img, ellipse_params
    
    parser = argparse.ArgumentParser(description='Draw an ellipse mask on an image or video frame.')
    parser.add_argument('--source', type=str, help='Path to image or video file')
    args = parser.parse_args()

    if args.source:
        if not os.path.exists(args.source):
            print(f"Error: File not found at {args.source}")
            return

        # Try loading as image first
        clean_img = cv2.imread(args.source)
        
        # If not an image, try loading as video
        if clean_img is None:
            cap = cv2.VideoCapture(args.source)
            ret, frame = cap.read()
            if ret:
                clean_img = frame
            cap.release()
            
        if clean_img is None:
            print(f"Error: Could not load image or video from {args.source}")
            return
    else:
        # Default black canvas
        clean_img = np.zeros((512, 512, 3), np.uint8)

    # Create a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_ellipse)

    print("Controls:")
    print("  Mouse: Click and drag to draw an ellipse")
    print("  Move: 'w' (Up), 's' (Down), 'a' (Left), 'd' (Right)")
    print("  Resize Height: 'i' (Increase), 'k' (Decrease)")
    print("  Resize Width: 'l' (Increase), 'j' (Decrease)")
    print("  'Space': Save the mask as 'ellipse_mask.png'")
    print("  'c': Clear the canvas")
    print("  'q': Quit")

    while True:
        # Start with a clean image each frame
        display_img = clean_img.copy()

        # Draw the ellipse if it exists
        if ellipse_params:
            cv2.ellipse(display_img, 
                        ellipse_params['center'], 
                        ellipse_params['axes'], 
                        ellipse_params['angle'], 
                        0, 360, (0, 255, 0), 2)

        cv2.imshow('image', display_img)
        k = cv2.waitKey(10) # Small delay to allow for smooth keyboard input

        if k == -1:
            continue
            
        # Quit
        if k & 0xFF == ord('q'):
            break
            
        # Save
        elif k & 0xFF == ord(' '):
            mask = np.zeros(clean_img.shape[:2], np.uint8)
            if ellipse_params:
                cv2.ellipse(mask, 
                            ellipse_params['center'], 
                            ellipse_params['axes'], 
                            ellipse_params['angle'], 
                            0, 360, 255, -1)
            cv2.imwrite('ellipse_mask.png', mask)
            print("Mask saved as 'ellipse_mask.png'")
            
        # Clear
        elif k & 0xFF == ord('c'):
            ellipse_params = None
            print("Canvas cleared")

        # Keyboard Controls for Ellipse Adjustment
        elif ellipse_params:
            cx, cy = ellipse_params['center']
            ax_h, ax_v = ellipse_params['axes']
            step = 2 # Pixels to move/resize per key press

            # Resize (IJKL)
            if k & 0xFF == ord('i'): # Taller
                ax_v += step
            elif k & 0xFF == ord('k'): # Shorter
                ax_v = max(1, ax_v - step)
            elif k & 0xFF == ord('l'): # Wider
                ax_h += step
            elif k & 0xFF == ord('j'): # Narrower
                ax_h = max(1, ax_h - step)
            
            # Move (WASD)
            elif k & 0xFF == ord('w'): # Up
                cy -= step
            elif k & 0xFF == ord('s'): # Down
                cy += step
            elif k & 0xFF == ord('a'): # Left
                cx -= step
            elif k & 0xFF == ord('d'): # Right
                cx += step

            ellipse_params['center'] = (cx, cy)
            ellipse_params['axes'] = (ax_h, ax_v)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
