import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# we now create a list that will get images automatically from the folder the it generate encoding for it & try to find
# it in our webcam we do this b/c if we have to write code for all image encoding then it will become messy

path = 'ImagesAttendance'
images = []
classNames = []  # for names of all images in list, grab list of images in this
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # it will gives us first name only i.e. Bill Gate not Bill Gates.jpg
print(classNames)


# now we are going to do encoding of images

def findEncodings(images):
    encodeList = []  # empty list having all encodings in the end
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        # print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])  # it append first element i.e. name in list
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# markAttandance('a')


encodeListKnown = findEncodings(images)  # list of known images
print('Encoding Complete Successfully')

# Step - 3, find matching b/w our encodings, and image for matching we use webcam

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # it will resize the original image to make the processing fast
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # now we find encoding for our webcam

    facesCurFrame = face_recognition.face_locations(imgS)  # in webcam we find multiple faces so that we find location
    # of our faces, then we send these locations to our encoding function
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # now find matches , compare all images in CurFrame (Current Frame) & compare it with encodeListKnown
    for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):  # it takes face Location (faceLoc) from facesCurFrame
        # one by one and then it takes encoding of faces from encodeCurFrame
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # list of faces
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0),cv2.FILLED)  # it has rectangle at the bottom & it shows name
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)  # when we find a face, we call this function & give it the name

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# NOTE : If a peron is already present then after that when we will bring their image/picture there result
# Attendance is not noted b/c it is already noted/marked i.e. it only mark attendance of left,
# whose Attendance is not recorded


# faceLoc = face_recognition.face_locations(imgElon)[0] # b/c we are sending only single img
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(faceLoc) provide dimension of image
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)  # to know where we detect the face

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# now we are at final step we have to compare the faces & find the distance b/w them i.e we
# #compare encoding of both img

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
