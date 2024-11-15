import cv2
import os
import time
import shutil
from tkinter import messagebox
import pymysql


def getTrainingData(window_name, camera_id, path_name, max_num,facedatapath):  # path_name is the image storage directory, and max_num is the number of images that need to be captured
    cv2.namedWindow(window_name)  # create a window
    cap = cv2.VideoCapture(camera_id)  # open camera
    classifier = cv2.CascadeClassifier('cv2_data/haarcascade_frontalface_alt.xml')  # load the built-in classifier of Opencv
    color = (0, 255, 0)  # The color of the face rectangle is green
    num = 0  # Record the number of stored images

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("no sign input ,please check your camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray image
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # Capturing face, writing the name of the captured image, using formatted string output
                image_name = '%s%d.jpg' % (path_name, num)
                image = frame[y:y + h, x:x + w]  # Save the face containing part of the current frame as an image, starting from the y bit and ending at y+h-1 bit
                cv2.imwrite(image_name, image)

                print(facedatapath) #Folder Path
                image_path = os.path.realpath(image_name)  #Obtain absolute path
                print(image_path)  #Image Path
                shutil.move(image_path, facedatapath)  # Move image files to the corresponding folder

                num += 1
                # Exit the loop if the specified maximum save quantity is exceeded
                if num > max_num:
                    break

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Draw a rectangular box
                font = cv2.FONT_HERSHEY_SIMPLEX  # Get built-in fonts
                cv2.putText(frame, ('%d' % num), (x + 30, y + 30), font, 1, (255, 0, 255),
                            4)  # Call the function to add a (x+30, y+30) rectangular box to the facial coordinate position to display how many facial images have been captured currently
        if num > max_num:
            break
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c == 27:  # Press the 'q' key to exit
            break

    cap.release()
    messagebox.showinfo("capture Finished", "Successfully captured face, welcome to the study room！")    # successful prompt
    cv2.destroyAllWindows()


#insert data into sql
def add_record(path_name):
    con = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='root',
        charset='utf8',
        database='face'
    )

    intime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    sql = "INSERT INTO user (name, intime, outtime) VALUES (%s, %s, %s)"
    cursor = con.cursor()

    try:
        cursor.execute(sql, (path_name, intime, 0))  # Using parameterized queries
        con.commit()  # Submit transaction
        print("Success to capture")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        con.close()



def read(n):
    na=n
    faceDataPath = 'faceData/'+na  # face data path
    max_num = 100  # Maximum number of faces captured
    faceDataPath_absPath = os.path.abspath(os.path.join(os.getcwd(), faceDataPath))  # absolute path
    print('Capturing face, writing', faceDataPath_absPath, 'folder')  # obtain face image
    if not os.path.exists(os.path.join(os.getcwd(), faceDataPath)):  # Is there a folder for storing facial data
        os.mkdir(os.path.join(os.getcwd(), faceDataPath))  # If it does not exist, create it

    #create sql record
    add_record(na)
    #Please note that the training_data-xx folder is located in the working directory of the program, along with the path to the # faceDataMath_absPath folder
    getTrainingData('getTrainData', 0, faceDataPath, max_num,faceDataPath_absPath)


