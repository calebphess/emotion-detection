from fer import FER
import cv2


# Pre-defined variables
USE_MTCNN = True # use MRCNN instead of Haar Cascade
TEST_IMAGE_PATH = "sample-images/man-happy.jpg"

# Load the emotion detector 
emotion_detector = FER(mtcnn=USE_MTCNN)

# Load a test image with opencv
test_image = cv2.imread(TEST_IMAGE_PATH)

# Analyze the image for faces and emotions
analysis = emotion_detector.detect_emotions(test_image)

# Print out the results
print(f"Number of faces: {len(analysis)}")

# Print emotion data for each face
for face_num, face in enumerate(analysis):
    emotions = face["emotions"] # get the full list of emotions
    primary_emotion = max(emotions, key=emotions.get) # get the highest confidence emotion
    primary_confidence = emotions[primary_emotion] # get the confidence of the primary emotion
    other_emotions = {} # holds list of other non-primary emotions

    # Loop through and add all non-empty emotions to the list of "other emotions"
    for emotion, confidence in emotions.items():
        if confidence > 0.0 and emotion != primary_emotion:
            other_emotions[emotion] = confidence


    print(f"----------------------------- Face {face_num} -----------------------------")
    print(f"Bounding Box: {face['box']}")
    print(f"Emotion: {primary_emotion}")
    print(f"Confidence: {primary_confidence}")
    print(f"Other Emotions : {other_emotions}")
    print(f"------------------------------------------------------------------")