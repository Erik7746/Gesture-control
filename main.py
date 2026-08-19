from src.config import Config
from src.camera import Camera, CameraError
import cv2

def main() -> None:
    config = Config()
    
    with Camera(config) as camera:
        while True:
            frame = camera.read()
            
            if config.mirror:
                frame = cv2.flip(frame, 1)
            
            cv2.imshow("Gesture Control", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' o ESC
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    raise SystemExit(main())