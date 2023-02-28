import cv2

# Read the input video
cap = cv2.VideoCapture("imgs/tmp_part_000.mp4")

# Get the video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("imgs/output_video.mp4", fourcc, fps, (width, height))

# Initialize the temporal mean frame for each channel
temp_mean_r = None
temp_mean_g = None
temp_mean_b = None

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Split the frame into its color channels
        b, g, r = cv2.split(frame)

        # Calculate the temporal mean for each channel
        if temp_mean_r is None:
            temp_mean_r = r.copy().astype("float32")
            temp_mean_g = g.copy().astype("float32")
            temp_mean_b = b.copy().astype("float32")
        else:
            cv2.accumulateWeighted(r, temp_mean_r, 0.5)
            cv2.accumulateWeighted(g, temp_mean_g, 0.5)
            cv2.accumulateWeighted(b, temp_mean_b, 0.5)

        # Subtract the temporal mean from each channel in the frame
        r_diff = cv2.subtract(r, cv2.convertScaleAbs(temp_mean_r))
        g_diff = cv2.subtract(g, cv2.convertScaleAbs(temp_mean_g))
        b_diff = cv2.subtract(b, cv2.convertScaleAbs(temp_mean_b))

        # Merge the deflickered color channels back into a BGR frame
        deflickered = cv2.merge((b_diff, g_diff, r_diff))

        # Write the deflickered frame to the output video file
        out.write(deflickered)

    else:
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
