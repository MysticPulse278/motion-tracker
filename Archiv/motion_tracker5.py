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
        self.events = []  # List to store separate bubble events
        self.current_event = []  # Temporary storage for current event

    @staticmethod
    def center_of_mass(binary_image):
        # calculate moments of binary image
        M = cv2.moments(binary_image)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return [cY, cX]
        return None

    def process_frame(self, frame):
        """Process a single frame and return the center of mass and if frame is black."""
        # Convert to grayscale and invert
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        
        # Threshold the image
        _, thresh = cv2.threshold(inv, 20, 255, 0)
        
        # Check if the frame is completely black (no bubble)
        is_black = not np.any(thresh)
        
        # Find center of mass
        com = self.center_of_mass(thresh)
        
        return thresh, com, is_black

    def track_motion(self):
        """Process the video and track motion, separating bubble events."""
        frame_number = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process the frame
            thresh, com, is_black = self.process_frame(frame)
            
            # Handle event separation
            if com is not None:
                position_with_frame = [frame_number] + com  # Store frame number with position
                self.trajectories.append(com)
                
                if is_black:
                    # End of bubble event
                    if self.current_event:
                        self.events.append(self.current_event)
                        self.current_event = []
                else:
                    # Ongoing bubble event
                    self.current_event.append(position_with_frame)

            # Create display frame
            display_frame = frame.copy()
            if com is not None:
                # Draw center of mass (green circle)
                cv2.circle(display_frame, 
                          (int(com[1]), int(com[0])), 
                          5, (0, 255, 0), -1)

            # Combine original and threshold
            combined = np.hstack((display_frame, 
                                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

            cv2.imshow('Motion Tracking', combined)
            
            frame_number += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Add the last event if it exists
        if self.current_event:
            self.events.append(self.current_event)

        self.cap.release()
        cv2.destroyAllWindows()

    def plot_trajectories(self):
            """Plot all motion events starting at the same time."""
        if not self.trajectories:
            print("No trajectories to plot!")
            return

        trajectories = np.array(self.trajectories)

        # Identify event segments
        events = []
        current_event = []
        
        for i, point in enumerate(trajectories):
            if point[0] == 0 and point[1] == 0:  # Black frame (no motion)
                if current_event:  # If there was an event, store it
                    events.append(current_event)
                    current_event = []
            else:
                current_event.append((i, point[0]))  # Store (frame, vertical position)

        if current_event:  # Add the last event if it wasn't followed by a black frame
            events.append(current_event)

        # Align and plot events
        plt.figure(figsize=(8, 6))

        for event in events:
            event = np.array(event)
            time = np.arange(len(event))  # Normalize time to start at 0
            vertical_pos = event[:, 1]
            plt.scatter(time, vertical_pos, s=5, alpha=0.5)  # Scatter plot each event

        plt.title("Aligned Motion Events")
        plt.xlabel("Normalized Time (frames)")
        plt.ylabel("Vertical Position (pixels)")
        plt.show()



def main():
    # Replace with your video path
    video_path = '/home/frederik/Videos/untitled.mov'
    
    cv2.namedWindow('Motion Tracking', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Motion Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        tracker = MotionTracker(video_path)
        tracker.track_motion()
        tracker.plot_trajectories()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
