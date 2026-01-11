import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

skip = 0
face_data = [] 
dataset_path = '/face_dataset'
file_name = input("Enter the name of the person: ")

while True:
    ret,frame = cap.read()

    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(grey_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    k = 1

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse = True)

    skip += 1

    for face in faces[:1]:
        x, y, w, h = face

        offset = 5
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv.resize(face_section, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))

        cv.imshow(str(k), face_section)
        k += 1
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv.imshow("faces", frame)

    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + file_name + '.npy', face_data)
print("Data Successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv.destroyAllWindows()

# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
# img = cv.imread('sachin.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()