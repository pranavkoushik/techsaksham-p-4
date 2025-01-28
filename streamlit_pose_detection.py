import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def process_image(image, pose):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect poses
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    return image_rgb, results.pose_landmarks is not None

def main():
    st.title("Human Pose Detection")
    st.write("Choose between uploading an image or using your webcam for pose detection!")

    # Create a select box for choosing the input method
    input_method = st.selectbox("Select Input Method", ["Upload Image", "Use Webcam"])

    if input_method == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Create a temporary file to save the uploaded image
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # Read the image
            image = cv2.imread(tfile.name)
            
            if image is not None:
                with mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5) as pose:
                    
                    # Process the image
                    processed_image, pose_detected = process_image(image, pose)
                    
                    # Display results
                    if pose_detected:
                        st.success("Pose detected!")
                    else:
                        st.warning("No pose detected in the image.")
                    
                    st.image(processed_image, channels="RGB", caption="Processed Image")
            else:
                st.error("Error reading the image. Please try another file.")

    else:  # Webcam option
        st.write("Click the button below to start the webcam. Press 'Q' in the opencv window to stop.")
        if st.button("Start Webcam"):
            cap = cv2.VideoCapture(0)
            
            with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
                
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        st.error("Failed to read from webcam")
                        break

                    # Process the frame
                    processed_image, _ = process_image(image, pose)
                    
                    # Display the frame
                    cv2.imshow('MediaPipe Pose Detection (Press Q to quit)', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
