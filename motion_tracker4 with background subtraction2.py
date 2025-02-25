import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

class MotionTracker:
    def __init__(self, video_path):
        """Initialize the motion tracker with video path."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.trajectories = []
        self.previous_gray = None
        self.erosion_kernel = np.ones((3, 3), np.uint8)  # Erosion kernel

    @staticmethod
    def center_of_mass(binary_image):
        """Calculate center of mass from binary image."""
        M = cv2.moments(binary_image)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]# signifikante Stellen?
            cY = M["m01"] / M["m00"]
        else:
            return [0, 0]
        return [cY, cX]

    def process_frame(self, frame):
        """Process frame with differencing and erosion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_gray is None:
            self.previous_gray = gray
            return np.zeros_like(gray), [0, 0]
        
        diff = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray
        blur =  cv2.GaussianBlur(diff, (13,13), 0)
        # Threshold and apply erosion
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, self.erosion_kernel, iterations=1)
        
        com = self.center_of_mass(thresh)
        return thresh, com

    def track_motion(self):
        """Process video and track motion."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                frame = frame[216:1139, 213:391]
            except:
                pass
            thresh, com = self.process_frame(frame)
            if com is not None:
                self.trajectories.append(com)

            if com is not None:
                cv2.circle(frame, 
                          (int(com[1]), int(com[0])),
                          5, (0, 255, 0), -1)

            combined = np.hstack((frame,
                                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
            cv2.imshow('Motion Tracking', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_trajectories(self, filename='trajectories.csv'):
        """Save trajectories to CSV file."""
        if not self.trajectories:
            print("No trajectories to save!")
            return
            
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame_number', 'x', 'y'])
            for frame_num, pos in enumerate(self.trajectories):
                writer.writerow([frame_num, pos[1], pos[0]])

    def plot_trajectories(self):
        """Plot motion trajectories."""
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
    video_path = '20250225_103327.mp4'
    cv2.namedWindow('Motion Tracking', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Motion Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        tracker = MotionTracker(video_path)
        tracker.track_motion()
        tracker.save_trajectories()  # Save trajectories to CSV
        tracker.plot_trajectories()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()