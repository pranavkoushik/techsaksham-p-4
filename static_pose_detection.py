import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Replace this with the path to your image
IMAGE_PATH = 'path_to_your_image.jpg'  # For example: 'C:/Users/YourName/Pictures/pose.jpg'

# For static images:
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not read image from {IMAGE_PATH}")
        print("Please check if the image path is correct")
        exit(1)

    image_height, image_width, _ = image.shape
    
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("No pose landmarks detected in the image")
        exit(1)

    # Print nose coordinates as an example
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    
    # Draw segmentation on the image.
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (192, 192, 192)  # gray
    annotated_image = np.where(condition, annotated_image, bg_image)
    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Display the result
    cv2.imshow('MediaPipe Pose Detection', annotated_image)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the annotated image
    output_path = 'pose_detection_result.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"\nAnnotated image saved as: {output_path}")
