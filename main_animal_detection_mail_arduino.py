import cv2
import numpy as np
import time
import yagmail
import serial
import sys

arduino = serial.Serial('COM4', 9600, timeout=.1)

# Load Yolo Object Data
net = cv2.dnn.readNetFromDarknet("yolov8.cfg", "yolov8.weights")
#save all the names in file o the list classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Input Video...")

video_capture = cv2.VideoCapture(0)

def mail_send():
    mail = 'sampletestmail.786@gmail.com';    
    password = 'vnzymrxlzmdyurch'
    # list of email_id to send the mail 
    li = [ "palanivinoth5513@gmail.com"]
    body = "Animal or Bird Detected...!"
    filename = "output.jpg"

    yag = yagmail.SMTP(mail, password)

    for dest in li:
        yag.send(
            to=dest,
            subject="Alert...! Bird/Animal Detected..",
            contents=body,
            attachments=filename,
        )
    print("Mail sent to all...!")
    time.sleep(1)


while True:
    # Capture frame-by-frame
    re,img = video_capture.read()
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "bird":
                print('Bird Detected..!')
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y + 30), font, 1.5, color, 1)
                time.sleep(0.5)
                arduino.write(b'1\r\n')
                time.sleep(0.5)
                cv2.imwrite('output.jpg', img)
                mail_send()
            elif label == "cow":
                print('Cow Detected..!')
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y + 30), font, 1.5, color, 1)
                time.sleep(0.5)
                arduino.write(b'2\r\n')
                time.sleep(0.5)
                cv2.imwrite('output.jpg', img)
                mail_send()
            elif label == "sheep":
                print('Sheep Detected..!')
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y + 30), font, 1.5, color, 1)
                time.sleep(0.5)
                arduino.write(b'3\r\n')
                time.sleep(0.5)
                cv2.imwrite('output.jpg', img)
                mail_send()
            elif label == "elephant":
                print('Elephant Detected..!')
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y + 30), font, 1.5, color, 1)
                time.sleep(0.5)
                arduino.write(b'4\r\n')
                time.sleep(0.5)
                cv2.imwrite('output.jpg', img)
                mail_send()
            elif label == "bear":
                print('Bear Detected..!')
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y + 30), font, 1.5, color, 1)
                time.sleep(0.5)
                arduino.write(b'5\r\n')
                time.sleep(0.5)
                cv2.imwrite('output.jpg', img)
                mail_send()
            else:
                print('No Animal...!')
                time.sleep(0.5)
                arduino.write(b'0\r\n')
                time.sleep(0.5)
                

    cv2.imshow("Image",cv2.resize(img, (800,600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
