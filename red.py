import cv2
import numpy as np
import pickle
from math import trunc
import matplotlib.pyplot as plt

pickle_in=open("bestnormalizedone.p","rb")
model=pickle.load(pickle_in)
cap=cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,600)


while True:
    _x,frame=cap.read()
    #cv2.imshow("orignal  one",frame)

    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowblue=np.array([50,50,50])
    highblue=np.array([70,255,255])
    bluemask=cv2.inRange(hsv_frame,lowblue,highblue)


    cv2.imshow("Preprocessed Image",bluemask)
    bluemask = cv2.resize(bluemask, (28, 28))
    #cv2.imshow("after resize", bluemask)

    bluemask=abs(bluemask-255)/255
    #cv2.imshow("inversion", bluemask)

    #cv2.imshow("" ,bluemask)

    bluemask=bluemask.reshape(-1,28,28,1)
    num=int(model.predict_classes(bluemask))
    pred = np.max(model.predict(bluemask))
    print(num, pred)
    cv2.putText(frame, str(num) + " with probability of " + str(trunc(pred*100)) +"%", (80, 80), cv2.FONT_ITALIC , 1, (0, 10,10 ), 3)
    cv2.imshow("Orignal Image", frame)



    if cv2.waitKey(1)&0xFF=='q':
        break

cap.release()
cv2.destroyAllWindows()
