import cv2
import numpy as np
import matplotlib.pyplot as plt

class MotionTracker:
    def __init__(self, video_path):
        """Initialize the motion tracker with video path."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.trajectories = []

    @staticmethod
    def center_of_mass(binary_image):
        """Calculate the center of mass of a binary image."""
        # Get image dimensions
        height, width = binary_image.shape
        binary_image = np.bitwise_not(binary_image)
        # Calculate the sum of all pixel values
        total_mass = np.sum(binary_image)/255
        print(total_mass)
        print(binary_image)
        if total_mass==0:
            return [0,0]

        # Calculate the weighted sum of pixel indices for x and y
        x_indices = np.arange(width)
        y_indices = np.arange(height)

        x_center = np.sum(binary_image * x_indices) / total_mass / 255
        y_center = np.sum(binary_image.T * y_indices) / total_mass / 255

        return [y_center, x_center]


    def process_frame(self, frame):
        """Process a single frame and return the center of mass."""
        # Convert to grayscale and invert
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        
        # Threshold the image
        _, thresh = cv2.threshold(inv, 20, 255, 0)
        
        # Find center of mass
        com = self.center_of_mass(thresh)
        
        return thresh, com

    def track_motion(self):
        """Process the video and track motion."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process the frame
            thresh, com = self.process_frame(frame)
            if com is not None:
                self.trajectories.append(com)
                #print(com)

            # Create side-by-side display
            display_frame = frame.copy()
            if com is not None:
                # Draw center of mass (green circle)
                cv2.circle(display_frame, 
                          (int(com[1]), int(com[0])), 
                          5, (0, 255, 0), -1)

            # Combine original and threshold
            combined = np.hstack((display_frame, 
                                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

            # Show the combined frame

            cv2.imshow('Motion Tracking', combined)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def plot_trajectories(self):
        """Plot the motion trajectories."""
        if not self.trajectories:
            print("No trajectories to plot!")
            return

        trajectories = np.array(self.trajectories)
        
        # Plot vertical position over time
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(trajectories[:, 0], label='Vertical Position')
        plt.title('Vertical Position Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Position (pixels)')
        plt.legend()

        # Plot velocity
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
    # Replace with your video path
    video_path = '/home/frederik/Videos/untitled.mp4'
    
    cv2.namedWindow('Motion Tracking',cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Motion Tracking',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    try:
        tracker = MotionTracker(video_path)
        tracker.track_motion()
        tracker.plot_trajectories()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
