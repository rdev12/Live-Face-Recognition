#%%
import cv2
import os
import numpy as np
from imutils import paths
import pickle
import face_recognition

imagePaths = list(paths.list_images('./datasets/rishikesh'))

#To rename the files to numbers

# for (i,img_file) in enumerate(imagePaths):
#     os.rename(img_file, str(i) + ".jpg")
knownEncodings = []

for (i,img_file) in enumerate(imagePaths):

    print("[INFO] encoding for image {}".format(img_file))

    img = cv2.imread(img_file)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="cnn")

    #128-d encodings:
    encodings = face_recognition.face_encodings(img, boxes)

    knownEncodings.append(encodings[0])

    # top, right, bottom, left = boxes[0]

    # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    # cv2.imshow('input', img)

    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()