#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.chdir('W:\YOLOv5')

import warnings
warnings.filterwarnings('ignore')


# In[3]:


import cv2
import numpy as np


# In[31]:


cap = cv2.VideoCapture('Cars.mp4')
cap.isOpened()


# In[47]:


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using cap.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using cap.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


# In[5]:


model = cv2.dnn.readNetFromONNX('yolov5s.onnx')


# In[6]:


# with open('coco.names') as f:
#     classes = [line.strip() for line in f.readlines()]
# print(classes)


with open('coco.names') as f:
    classes = f.read().strip().split('\n')
print(classes)


# In[7]:


colors = np.random.randint(0, 255, (len(classes), 3))
colors.shape


# In[8]:


font = cv2.FONT_HERSHEY_SIMPLEX


# In[49]:


while cap.isOpened():
    
    ret, frame = cap.read() # frame shape = (360, 640, 3)
    if frame is None: break
    
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(640, 640),
                                 mean=[0, 0, 0], swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward()[0] # outputs shape = (1, 25200, 85)
    
    cls_idxs, confs, bboxes = [], [], []
    
    height, width = frame.shape[:-1] 
    scaled_x, scaled_y = width/640, height/640
    
    for bbox in outputs:
        conf = bbox[4]
        
        if conf>0.5:
            scores = bbox[5:]
            idx = np.argmax(scores)
            
            if scores[idx] > 0.5:
                confs.append(conf)
                cls_idxs.append(idx)
                
                cx, cy, w0, h0 = bbox[:4]
                x = int((cx - w0/2)*scaled_x)
                y = int((cy - h0/2)*scaled_y)
                w1 = int(w0 * scaled_x)
                h1 = int(h0 * scaled_y)
                
                bboxes.append([x, y, w1, h1])
    
    indices = cv2.dnn.NMSBoxes(bboxes, confs, 0.7, 0.2)
    
    assert len(bboxes)==len(cls_idxs)==len(confs)
        
    for i in indices:
        x, y, w, h = bboxes[i]
        label = classes[cls_idxs[i]]
        pc = confs[i]
        
        text = label+': ' + '{:.3f}'.format(pc)
        color = [int(c) for c in colors[cls_idxs[i]]]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x+1, y-4), font, 0.5, color)
    
    cv2.imshow('YOLOv5', frame)
    if cv2.waitKey(10) == ord('q'):
        break

cv2.destroyAllWindows()

