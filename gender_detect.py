
import cv2
import numpy as np


GENDER_MODEL = r"C:\Users\HP\OneDrive\Desktop\gender recognition\deploy_gender.prototxt"


GENDER_PROTO = r"C:\Users\HP\OneDrive\Desktop\gender recognition\gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

GENDER_LIST = ['Male', 'Female']

FACE_PROTO = r"C:\Users\HP\OneDrive\Desktop\gender recognition\deploy.prototxt.txt"

FACE_MODEL = r"C:\Users\HP\OneDrive\Desktop\gender recognition\res10_300x300_ssd_iter_140000_fp16.caffemodel"


face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)


frame_width = 1920
frame_height = 1080


def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    

    faces = []
    

    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]
  
    if width is None and height is None:
        return image


    if width is None:
      
        r = height / float(h)
        dim = (int(w * r), height)
 
    else:
       
        r = width / float(w)
        dim = (width, int(h * r))
   
    return cv2.resize(image, dim, interpolation = inter)


def predict_gender():
    """Predict the gender of the faces showing in the image"""
    
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
     
        frame = img.copy()
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        # predict the faces
        faces = get_faces(frame)
        

        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
         
            blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
                227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
           
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence_score = gender_preds[0][i]
 
            label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
       
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
         
            optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
            box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            # Label processed image
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)

       
        cv2.imshow("Gender Estimator", frame)
        if cv2.waitKey(1) == ord("q"):
            break



    
    cv2.destroyAllWindows()





if __name__ == '__main__':
    predict_gender()