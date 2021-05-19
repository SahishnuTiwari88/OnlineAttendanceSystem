import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImageBasic/Elon Musk.jpg') # it provides us function to load images
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0] # b/c we are sending only single img
encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(faceLoc) provide dimension of image
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)  # to know where we detect the face

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# now we are at final step we have to compare the faces & find the distance b/w them i.e we
# #compare encoding of both img

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest) # gives distance, lower the distance better the match
print(results,faceDis)  # if encoding is same i.e. of same person it gives true as result
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)
