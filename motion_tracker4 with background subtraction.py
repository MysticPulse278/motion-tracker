import cv2
import numpy as np
import matplotlib.pyplot as plt

class MotionTracker:
    def __init__(self, video_path):
        """Initialize the motion tracker with video path and compute background."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.frame_count)
        self.trajectories = []
        self.background = self.compute_background()
        
    def compute_background(self):
        """Compute time-averaged background from all video frames."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        background = None
        frame_count = 0 
      
        
        while True:
            ret, frame = self.cap.read()
            try:
                frame = frame[216:1139, 313:491]
            except:
                pass
            print(frame_count)
            if not ret:
                break
                
            # Convert to grayscale and accumulate
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if background is None:
                background = gray
            else:
                background += gray
            frame_count += 1

        # Calculate average and reset video
        background = (background / frame_count).astype(np.uint8)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cv2.imwrite("background.png", background)
        return background

    @staticmethod
    def center_of_mass(binary_image):
        """Calculate center of mass from binary image."""
        M = cv2.moments(binary_image)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            return [0, 0]
        return [cY, cX]

    def process_frame(self, frame):
        """Process a single frame with background subtraction."""
        # Convert to grayscale and subtract background
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.background)
        
        # Threshold the difference
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        
        # Find center of mass
        com = self.center_of_mass(thresh)
        return thresh, com

    def track_motion(self):
        cv2.namedWindow('Motion Tracking', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Motion Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        """Process the video and track motion."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                frame = frame[216:1139, 313:491]
            except:
                pass

            # Process the frame with background subtraction
            thresh, com = self.process_frame(frame)
            if com is not None:
                self.trajectories.append(com)

            # Create visualization
            display_frame = frame.copy()
            if com is not None:
                cv2.circle(display_frame, 
                          (int(com[1]), int(com[0])),
                          5, (0, 255, 0), -1)

            combined = np.hstack((display_frame,
                                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
            cv2.imshow('Motion Tracking', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def plot_trajectories(self):
        """Plot the motion trajectories."""
        if not self.trajectories:
            print("No trajectories to plot!")
            return

        trajectories = np.flip(np.array(self.trajectories), axis=0)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(121)
        plt.plot(trajectories[:, 0], label='Vertical Position')
        plt.title('Vertical Position Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Position (pixels)')
        plt.legend()

        plt.subplot(122)
        velocity = np.diff(trajectories[:, 0])
        plt.plot(velocity, label='Vertical Velocity')
        plt.title('Vertical Velocity Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Velocity (pixels/frame)')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    video_path = '20250225_103222.mp4'
    
    
    try:
        tracker = MotionTracker(video_path)
        tracker.track_motion()
        tracker.plot_trajectories()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()