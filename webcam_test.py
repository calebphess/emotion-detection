from fer import FER
import cv2


# Pre-defined variables
USE_MTCNN = True                # use MRCNN instead of Haar Cascade
NUM_FRAMES = 20                 # number of frames to wait between detecting faces and emotions
VIDEO_DEVICE = 0                # which video device to use
DETECTION_COLOR = (36,255,12)   # color of detection boxes and text
FONT_SIZE = 0.5                 # detection label font size

# Load the emotion detector 
emotion_detector = FER(mtcnn=USE_MTCNN)

# Load the default webcam into opencv
video = cv2.VideoCapture(VIDEO_DEVICE)

# Limits the amount of frames that the output is run on
frame_count = 0

# Holds the detection data to draw on the screen
detections = []

while(True):
    # Get the frame
    ret, frame = video.read()

    # Only process faces and emotions every X frames specified above
    if frame_count == NUM_FRAMES:

        # Analyze the image
        analysis = emotion_detector.detect_emotions(frame)
        
        # Clear out old detections
        detections = []

        # Print out the results
        print(f"Number of faces: {len(analysis)}")

        # Process emotion data for each face
        for face_num, face in enumerate(analysis):
            emotions = face["emotions"]                         # get the full list of emotions
            primary_emotion = max(emotions, key=emotions.get)   # get the highest confidence emotion
            primary_confidence = emotions[primary_emotion]      # get the confidence of the primary emotion
            other_emotions = {}                                 # holds list of other non-primary emotions

            # Loop through and add all non-empty emotions to the list of "other emotions"
            for emotion, confidence in emotions.items():
                if confidence > 0.0 and emotion != primary_emotion:
                    other_emotions[emotion] = confidence

            # Add the data to the detections list
            detection = {'emotion': primary_emotion, 'confidence': primary_confidence, 'box': face['box']}
            detections.append(detection)

            # Print data to the console
            print(f"----------------------------- Face {face_num} -----------------------------")
            print(f"Bounding Box: {face['box']}")
            print(f"Emotion: {primary_emotion}")
            print(f"Confidence: {primary_confidence}")
            print(f"Other Emotions : {other_emotions}")
            print(f"------------------------------------------------------------------")

        frame_count = 0
    else:
        frame_count += 1

    # Loop through each detection
    # Draw the detection boxes, emotion and confidence
    for detection in detections:
        x1 = detection['box'][0]
        y1 = detection['box'][1]
        height = detection['box'][2]
        width =  detection['box'][3]

        color = ()

        label = f"{detection['emotion']} - {detection['confidence']}"

        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), DETECTION_COLOR, 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, DETECTION_COLOR, 2)

    # Display the frame
    cv2.imshow('Webcam Emotion Detection Test', frame)

    # Use the 'q' button to stop the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break