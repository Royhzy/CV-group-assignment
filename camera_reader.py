import cv2
import time
from tkinter import messagebox
import pymysql
from datetime import datetime
from FaceRecognition.face import Ui_Form
from model_train import Model
from data_input import read_name_list
import math
from tkinter import messagebox, Tk  # root windows


#update data into sql
def ud_out_record(path_name):
    con = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='root',
        charset='utf8',
        database='face'
    )

    # record time
    outtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    sql = "UPDATE user SET outtime = %s WHERE name = %s"
    cursor = con.cursor()

    try:
        cursor.execute(sql, (outtime, path_name))  # Using parameterized queries
        con.commit()  # Submit transaction

        sql2 = "select intime from user where name = '%s'" % path_name
        cursor = con.cursor()
        cursor.execute(sql2)
        intime_result= cursor.fetchone()
        intime = intime_result[0]
        #print("intime",intime)

        # Convert a string to datetime
        time_format = "%Y-%m-%d %H:%M:%S"
        time1 = datetime.strptime(outtime, time_format)
        time2 = datetime.strptime(intime, time_format)

        delta_time = time1 - time2
        print('delta_time', delta_time)

        total_hours = delta_time.total_seconds() / 3600  # transform to hours
        rounded_hours = math.ceil(total_hours)  # carry bit

        payment = rounded_hours * 4

        show_info_message(path_name, delta_time, payment)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        con.close()


def show_info_message(path_name, delta_time, payment):
    root = Tk()  # create a root window
    root.withdraw()  # Hide root window

    message = f"Recognized name: {path_name}，Enter time: {delta_time}, Required pay: ${payment}"  # successful prompt
    messagebox.showinfo("Successfully recognized name", message)  # Display message box

    root.destroy()  # Destroy the root window to return to the main



#face recognize
def read():
    result_count = 0      # recognized counts
    result_last = 0       # recognized result
    faceData_file_path = './faceData/'
    cap = cv2.VideoCapture(0)

    # Read the names of subfolders under the dataset
    name_list = read_name_list(faceData_file_path)
    model = Model()
    model.load()

    cascade_path = "cv2_data/haarcascade_frontalface_alt.xml"  # OpenCV face detection classifier face detector
    cascade = cv2.CascadeClassifier(cascade_path)  # Load classifier

    while True:
        ret, frame = cap.read()  # Obtain camera images
        if not ret:
            print("no sign input, please check your camera")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray image

        # Detected faces
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))

        if len(facerect) > 0:  # If detected faces
            print(name_list)
            print('face detected')
            color = (0, 0, 255)  # opencv B-G-R (red)

            for rect in facerect:  # Traverse the detected faces
                # Obtain the x and y coordinates of the upper left corner of the face on the image, as well as the width and height of the face
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height + 10, x - 10: x + width + 10]  # Obtain facial images
                cv2.rectangle(frame, (x - 10, y - 10), (x + width + 10, y + height + 10), color, 2)  # draw rectangle

                result = model.predict(image)
                if result_last == result:     # compare with last result,if the same as last result,result count add 1
                    result_count += 1
                    print('resultcount', result_count)
                else:
                    result_last = result
                    result_count = 1    # Reset counts to 1 because this modification is the first time the data is recognized
                    print('result————count', result_count)

                print('PersonName:', name_list[result])
                cv2.putText(frame, name_list[result],
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0), 2)

                if result_count >= 20:
                    ud_out_record(name_list[result])  # select sql record
                    return   # If here break,system do not stop

        # Display images after face detection and processing
        cv2.imshow('face_recognition', frame)  # Display the processed image

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press the 'q' key to exit
            break


    # Release the camera
    cap.release()
    cv2.destroyAllWindows()




