from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.layers import TFSMLayer
import numpy as np
import imutils
import cv2
from imutils.video import VideoStream

# Low confidence threshold for face detection
lowConfidence = 0.75

def detectAndPredictMask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > lowConfidence:
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract the face from the frame
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs, preds)

# Paths for face detection model and mask detection model
prototxtPath = r"models/deploy.prototxt"
weightsPath = r"models/res10_300x300_ssd_iter_140000.caffemodel"

# Load the face detection model (OpenCV DNN)
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the mask detection model using TFSMLayer for TensorFlow SavedModel
maskModelPath = "models/mask_detector.model"
maskNet = TFSMLayer(maskModelPath, call_endpoint="serving_default")

# Start video stream
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    
    # Detect faces and predict mask status
    (locs, preds) = detectAndPredictMask(frame, faceNet, maskNet)
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        if label == "Mask":
            print("Mask Detected")
        else:
            print("Mask Not Detected")

        # Display label and confidence percentage
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # Show the output frame
    cv2.imshow("Press q to quit", frame)
    
    # Wait for key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()
