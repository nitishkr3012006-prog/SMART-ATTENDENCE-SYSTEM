import numpy as np
import cv2 as cv 
import os

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    distances = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        distances.append((d, iy))
    distances = sorted(distances)[:k]
    labels = np.array(distances)[:, 1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataset_path = './face_dataset/'

face_data = []
labels = [] 
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)       

font = cv.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grey_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse = True)

    for face in faces[:1]:
        x, y, w, h = face

        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        pred_name = names[int(out)]
        cv.putText(frame, pred_name, (x, y-10), font, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv.imshow("faces", frame)

    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv.destroyAllWindows()