import sys
import os

import cv2 as cv
import numpy as np

recognizer = cv.face.LBPHFaceRecognizer_create()
face_cascade = cv.CascadeClassifier('haar_frontalface_def.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        id =int(os.path.split(image_path)[-1].split(".")[0])

        faces = face_cascade.detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids 


class Application:
    def __init__(self):
        self.path_to_dataset = 'dataset'
        faces, ids = get_images_and_labels(self.path_to_dataset)
        recognizer.train(faces, np.array(ids))
        recognizer.write('trainer.yml')

    def face_recognition(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30,30))
        
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            id_predictable, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if (confidence < 100):
                if (id_predictable == 1):
                    cv.putText(frame, f"ID: Putin Conf: {confidence}",
                            (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame
    
    def run(self):
        cap = cv.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            frame = self.face_recognition(frame);
            cv.imshow("Application", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

Application().run()

