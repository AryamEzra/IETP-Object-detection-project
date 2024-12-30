import os
import cv2
import urllib.request
import numpy as np
import pyttsx3

# Ensure the correct working directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Replace the URL with the IP camera's stream URL
url = 'http://192.168.144.133/cam-hi.jpg'  # Replace with your ESP32 IP address

# YOLO configuration
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = os.path.join(script_dir, 'coco.names.txt')  # Use the correct file name

# Debug prints
print(f"Current working directory: {script_dir}")
print(f"coco.names.txt file path: {classesfile}")

classNames = []

# Open the classes file using the absolute path
try:
    with open(classesfile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"File not found: {classesfile}")
    exit()

modelConfig = os.path.join(script_dir, 'yolov3.cfg')
modelWeights = os.path.join(script_dir, 'yolov3.weights')
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize text-to-speech engine
engine = pyttsx3.init()

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    detected_objects = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                detected_objects.append(classNames[classId])
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Announce detected objects
    if detected_objects:
        detected_objects = list(set(detected_objects))  # Remove duplicates
        try:
            text = ", ".join(detected_objects)
            engine.say(text)  # Speak the text immediately
            engine.runAndWait()  # Wait until speaking is done
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

# Create a named window
cv2.namedWindow("Live Cam Testing", cv2.WINDOW_AUTOSIZE)

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the IP camera stream is opened successfully
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()

# Read and display video frames with YOLO detection
while True:
    try:
        # Read a frame from the video stream
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        
        # YOLO object detection
        blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img)
        
        # Display the image with detections
        cv2.imshow('Live Cam Testing', img)
        
        # Exit loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error reading the frame: {e}")
        break

cap.release()
cv2.destroyAllWindows()
