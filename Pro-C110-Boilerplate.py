import cv2
import numpy as np
import tensorflow as tf
  
model = tf.keras.models.load_model('keras_model.h5')

vid = cv2.VideoCapture(0)
  
while(True):
      
   
    ret, frame = vid.read()
  
    check,frame = vid.read()

    img=cv2.resize(frame,(224,224)) 

    test_image=np.array(img,type=np.float32)
    test_image=np.expand.dims(test_image,axis=0)
    

    normalized_image=test_image/255.0

    prediction=model.predict(normalized_image)

    print("prediction:",prediction)





    cv2.imshow('result', frame)
      
   
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  

vid.release()


cv2.destroyAllWindows()