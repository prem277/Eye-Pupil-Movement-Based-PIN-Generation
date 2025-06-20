import cv2
import numpy as np
import tensorflow as tf
import time

# Load the pre-trained model (ensure the path is correct)
model_path = "D:\Projects\PIN Generation\pupil_detection_model.keras"
model = tf.keras.models.load_model(model_path)

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_image_for_prediction(img):
    """
    Preprocess the input image for prediction.
    """
    eyes = detect_eyes(img)
    if len(eyes) == 0:
        print("No eyes detected")
        return None

    x, y, w, h = eyes[0]  # Use the first detected eye
    img = img[y:y+h, x:x+w]  # Crop the eye region

    # Show the cropped eye region
    cv2.imshow('Cropped Eye', img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def detect_eyes(frame):
    """
    Detect eyes in the input frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected eyes: {len(eyes)}")
    
    # Draw rectangles around detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    # Show the frame with detected eyes
    cv2.imshow('Detected Eyes', frame)
    return eyes

def predict_movement(frame):
    """
    Predict movement based on the detected eye region.
    """
    preprocessed_image = preprocess_image_for_prediction(frame)
    if preprocessed_image is None:
        return None

    try:
        # Get raw prediction
        prediction = model.predict(preprocessed_image)
        print(f"Raw prediction: {prediction}")
        
        movement = np.argmax(prediction)
        if movement == 1:
            print("Prediction: Right ( - )")
        elif movement == 0:
            print("Prediction: Left ( . )")
        else:
            print(f"Prediction: Unclear movement: {movement}")
        return movement
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    """
    Main function to capture video feed and detect eye movements.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not opened.")
        return

    last_movement = None
    consistent_time = 0
    morse_code = ""
    morse_code_map = {0: '.', 1: '-'}
    duration_threshold = 2  # seconds
    max_gap = 1  # seconds allowed between detections

    print("Please move your eyes to enter the pattern.")

    start_time = time.time()

    while len(morse_code) < 3:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the raw frame for debugging
            cv2.imshow('Camera Feed', frame)

            movement = predict_movement(frame)

            current_time = time.time()

            if movement is not None:
                if movement == last_movement:
                    consistent_time += current_time - start_time
                else:
                    consistent_time = 0  # Reset if movement changes
                    last_movement = movement

                if consistent_time >= duration_threshold:
                    code = morse_code_map[movement]
                    morse_code += code
                    direction = 'Right' if movement == 1 else 'Left'
                    print(f"Detected movement: {direction} -> Code: {code}")
                    consistent_time = 0
                    last_movement = None  # Reset after recording a code

            else:
                # Allow gaps but reset timer if gap exceeds max_gap
                if current_time - start_time > max_gap:
                    consistent_time = 0
                    last_movement = None

            start_time = current_time  # Update start time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error during processing: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f'Generated Morse Code: {morse_code}')

    predefined_morse_pin = "---"  # Example Morse PIN

    if morse_code == predefined_morse_pin:
        print("Access Granted")
    else:
        print("Access Denied")

if __name__ == "__main__":
    main()
