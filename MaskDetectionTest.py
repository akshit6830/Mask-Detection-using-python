#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.models import load_model 
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox 
import smtplib
import threading 

print("gsdfg")
model = load_model('model_100.h5')

haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

text_dict = {0 : 'Mask ON' , 1 : "No Mask"}
color_dict = {0 : (0,255,0) , 1  : (0,0,255) }

def ckeckMaskMsg():
    root = Tk()
    root.withdraw()
    while cap.isOpened():
        if label != None :
            if label == 1 :
                messagebox.showinfo("MASK PLEASE !" , "Please Wear Mask for your Safety and for others too")
    root.mainloop()

t1 = threading.Thread(target=ckeckMaskMsg)
t1.start()

while cap.isOpened():
    
    ret , img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray)
       
    for x,y,w,h in faces : 
        
        face_img = gray[y:y+w, x:x+w]
        resized_img = cv2.resize(face_img , (112 , 112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img , (1,112,112,1))
        result = model.predict(reshaped_img)
        
        label = np.argmax(result , axis=1)[0]
        
        cv2.rectangle(img , (x,y) , (x+w , y+h) , color_dict[label] , 2 )
        cv2.rectangle(img , (x,y-40) , (x+w , y) , color_dict[label] , -1 )
        cv2.putText(img , text_dict[label] , (x,y-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.8 , (0,0,0) , 2)
                    
                        
    cv2.imshow("results" , img )
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q') :
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




