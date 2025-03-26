# CV-group-assignment: 

# README

YouTube Link: [https://youtu.be/5OtfIvx2wzE](https://www.youtube.com/watch?v=5OtfIvx2wzE)

# Study Room Face Recognition System

This project aims to utilize Python and OpenCV for facial recognition in an autonomous study room scenario. The system tracks user attendance, records entry and exit times, and calculates the duration of stay using a face recognition system based on computer vision techniques.


## Group Conception

After discussions among group members, we explored various applications of computer vision (CV) in different industries, such as medical image analysis, parking systems, face recognition, and self-checkout systems. We ultimately chose to focus on a study room attendance system using face recognition. This system aims to improve efficiency by reducing manual operations, enhancing accuracy, and improving security by ensuring only registered users can access the study room.


## System Components

1. **Authentication Module**: Allows administrators to log into the system by entering their name and account number.
2. **Face Detection**: Captures real-time input from the camera and detects faces using OpenCV.
3. **Model Training**: Trains the neural network to recognize faces from the collected dataset.
4. **Face Enrollment**: Enrolls faces into the system's database.
5. **Time Record**: Calculates total time spent by each individual and displays the cost.

## Tools Overview

1. **Python**

2. **OpenCV**

3. **TensorFlow**

4. **Keras**

5. **PyQt5**


## UI
![1731668410647(1)](https://github.com/user-attachments/assets/e7fa8ad9-d4a6-499f-b73d-8e7f598075e1)

![1731668410647(1)](https://github.com/user-attachments/assets/df96826a-e4f6-4dfd-978c-851332e36959)

## Database
![1731670028062](https://github.com/user-attachments/assets/9c38f35b-6608-4109-a8aa-20fffa41e3b0)


## Neural Network Training Results
After numerous training attempts, it was observed that the loss and validation values did not change around 40,000 training iterations. Therefore, the training was set to 40,000 iterations. 

![simple_nn_plot_acc](https://github.com/user-attachments/assets/ead9743d-6ef1-4b01-9a7f-c63e0196f722)

Figure shows the ratio of training iterations to loss values. It demonstrates that the loss value decreases as the number of training iterations increases, meaning that the neural network starts to converge.

![simple_nn_plot_loss](https://github.com/user-attachments/assets/0b69bdb8-8a96-4858-b326-98998dc249c8)


## Real Face Testing
The display method of face detection is to frame the face on the input video, and display the recognized face name next to the frame. below figures are the running results.
（Facial Mosaic Protection for Portrait Rights）

![1731670787305](https://github.com/user-attachments/assets/304dab53-fb08-4d10-ac9b-8a40602deec3)

![image](https://github.com/user-attachments/assets/c61d989f-8103-4164-80c4-50cd0e729644)



## References
1. Shehu, V., & Dika, A. (2010). Using real-time computer vision algorithms in automatic attendance management systems. ITI 2010, 32nd International Conference on Information Technology Interfaces.
2. Dai Yuqi, Wang Xin, & Yao Naiwen. (2022). Analysis of the consumption phenomenon of youth paid study rooms. Chinese Youth Social Sciences.
3. Mohammed, K., Tolba, A. S., & Elmogy, M. (2018). Multimodal student attendance management system (MSAMS). Ain Shams Engineering Journal.
4. Lafta, N. A., & Abbood, Z. A. A. (2024). Comprehensive Review and Comparative Analysis of Keras for Deep Learning Applications: A Survey on Face Detection Using CNNs. International Journal of Religion.
5. Lopes, A. T., et al. (2017). Facial expression recognition with CNNs: coping with few data and training sample order. Pattern Recognition.
6. Tan, S. J. (2018). Facial recognition-based attendance monitoring system for educational institutions.
7. Oliphant, T. E. (2007). Python for scientific computing. Computing in Science & Engineering.
8. Minichino, J., & Howse, J. (2015). Learning OpenCV 3 Computer Vision with Python.
9. Rampasek, L., & Goldenberg, A. (2016). TensorFlow: biology’s gateway to deep learning?. Cell Systems.
10. Wongsuphasawat, K., et al. (2017). Visualizing dataflow graphs of deep learning models in TensorFlow. IEEE Transactions on Visualization and Computer Graphics.
11. Moolayil, J., & John, S. (2019). Learn Keras for Deep Neural Networks.
12. Kravets, A. G., et al. (2021). Relevant image search method when processing a patent array.
13. Shah, M. (1997). Fundamentals of computer vision.
14. Sun, Y., Wang, X., & Tang, X. (2013). Hybrid deep learning for face verification.
15. Michel, P., & El Kaliouby, R. (2003). Real-time facial expression recognition in video using support vector machines.
16. Khan, S., et al. (2020). Real-time automatic attendance system for face recognition using OpenCV.17. Santos, C. F. G. D., & Papa, J. P. (2022). Avoiding overfitting in CNNs: A survey on regularization methods.



