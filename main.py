# Add the directory to the search path
import sys
sys.path.append('src')

from mosse.mosse import *
import glob2
import time

if __name__ == "__main__":

    # Specify the image sequence directory and file pattern
    # Ex : Suv, ReadTeam, David 
    image_sequence_dir = "data/RedTeam/img/"
    image_file_pattern = "*.jpg"

    # Get a list of image file paths
    image_paths = glob2.glob(image_sequence_dir + image_file_pattern)

    # Sort the image file paths
    image_paths.sort()

    # Read the initial frame
    initial_frame = cv2.imread(image_paths[0])
    initial_frame_gray = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)

    # Select the ROI in the initial frame
    roi = cv2.selectROI("Initial Frame", initial_frame, fromCenter=False, showCrosshair=True)

    tracker = MosseTracker()
    tracker.init(initial_frame_gray, roi)
    
    # Process each frame in the image sequence
    for image_path in image_paths:
        # Read the current frame
        frame = cv2.imread(image_path)
        frame_copy = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        # Update
        bbox = tracker.update(frame_copy)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 2)
        print("PSR : ", tracker.psr)

        # Display the current frame with the ROI
        cv2.imshow("Video", frame)

        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()