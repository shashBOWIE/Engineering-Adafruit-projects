#Physical Therapy Motion Capture script
#engineering 4
# James Bowie HS
#Julius Rivera --- October 28 2022
#FINISHED Julius Rivera and Shash K --- December 14 2022

#importing libraries

import cv2
import csv
import time
#anything using numpy is instead np.whatever
import numpy as np

#mouseClick function finds HSV values of a clicked pixel on the screen
def mouseClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = frame_hsv[y,x]
        print("CLICK: x = ", x, "y = ", y)
        print("Hue = ", hsv[0], "Saturation = ", hsv[1], "Value = ", hsv[2])

#centroidFinder function returns the x and y values of the center of a color region
def centroidFinder(input_image, output_image):
    moments = cv2.moments(input_image)
    if moments["m00"] == 0:
        centroid_x = 1
        centroid_y = 1
    else:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    cv2.circle(output_image, (int(centroid_x), int(centroid_y)), 5, (255,255,255), -1)
    return int(centroid_x), int(centroid_y)

#angle function is the dot product of two distance vectors, returns magnitude of dot product vector
def angle(a, b):
    if np.linalg.norm(a) != 0 and np.linalg.norm(b) != 0:
        arccos_argument = np.dot(a, b)/ (np.linalg.norm(a) * np.linalg.norm(b))
        #arccos_argument makes it not return undefined
        if  arccos_argument >= -1 and arccos_argument <= 1:
            return np.degrees(np.arccos((np.dot(a, b))/ (np.linalg.norm(a) * np.linalg.norm(b))))

#videoCapture(0) for internal camera, videoCapture(1) for external camera
cap = cv2.VideoCapture(0) #argument is webcam number

cv2.namedWindow("WebCam")
#Hue Sliders are made to adjust for lighting
#you can use mouseClick to find the Hue value and adjust colors accordingly
cv2.namedWindow("Hue Sliders")
cv2.setMouseCallback("WebCam", mouseClick, param = None)

#create colors to look for
#base HSV values:

#Green Escarpment is correlated to color 2, is not actually used on the wearable
#hue sliders are made for each color to adjust for lighting conditions
#Hyperactive color:
c1_hueOG = 9
cv2.createTrackbar("Hue 1", 'Hue Sliders', 0, 5, lambda x: None)
cv2.setTrackbarMin("Hue 1", 'Hue Sliders', -5)
c1_hue = c1_hueOG + cv2.getTrackbarPos("Hue 1", 'Hue Sliders')
c1_upper = [c1_hue + 3, 255, 255]
c1_lower = [c1_hue - 3, 160, 160]

#Green Escarpment color:
c2_hueOG = 91
cv2.createTrackbar("Hue 2", "Hue Sliders", 0, 5, lambda x: None)
cv2.setTrackbarMin("Hue 2", 'Hue Sliders', -5)
c2_hue = c2_hueOG + cv2.getTrackbarPos("Hue 2", 'Hue Sliders')
c2_upper = [c2_hue + 4, 255, 215]
c2_lower = [c2_hue - 4, 140, 55]

#Liming color:
c5_hueOG = 30
cv2.createTrackbar("Hue 3", "Hue Sliders", 0, 5, lambda x: None)
cv2.setTrackbarMin("Hue 3", 'Hue Sliders', -5)
c5_hue = c5_hueOG + cv2.getTrackbarPos("Hue 3", 'Hue Sliders')
c5_upper = [c5_hue + 6, 200, 255]
c5_lower = [c5_hue - 6, 70, 100]

#Thalia Pink 178 162
c6_hueOG = 170
cv2.createTrackbar("Hue 4", "Hue Sliders", 0, 5, lambda x: None)
cv2.setTrackbarMin("Hue 4", 'Hue Sliders', -5)
c6_hue = c6_hueOG + cv2.getTrackbarPos("Hue 4", 'Hue Sliders')
c6_upper = [c6_hue + 8, 210, 255]
c6_lower = [c6_hue - 8, 155, 80]



kernel = np.ones((5,5) , np.uint8)

#pressing 'w' at any point will stop the code
print("stop code by pressing 'w' key")
print("press 'e' to start data acqusition")

#data_filename = input("data code title?")
#cv2.imwrite(data_filename, 
f = open(r'C:\Users\juliu\PythonCode\data1test.csv', 'w')

# create the csv writer
writer = csv.writer(f)

header = ['Time', 'Angle']
    
writer.writerow(header)

#while loop to ensure the code continues looping until the code is stopped by pressing 'w' key

#sets keypressed so the code doesn't break
keypressed = 0
while keypressed != ord('e'):
    if keypressed == ord('w'):
        cv2.destroyAllWindows()
        cap.release()
    #initialize the webcam

    #frame can be changed, "ret" is unknown and can not be changed
    #frame is the name of the image coming from the webcam
    ret, frame = cap.read()

    #create HSV image
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #allows colors to change based on slider input
    c1_hue = c1_hueOG + cv2.getTrackbarPos("Hue 1", 'Hue Sliders')
    c1_upper = [c1_hue + 3, 255, 255]
    c1_lower = [c1_hue - 3, 160, 160]

    c2_hue = c2_hueOG + cv2.getTrackbarPos("Hue 2", 'Hue Sliders')
    c2_upper = [c2_hue + 4, 255, 215]
    c2_lower = [c2_hue - 4, 140, 55]

    c5_hue = c5_hueOG + cv2.getTrackbarPos("Hue 3", 'Hue Sliders')
    c5_upper = [c5_hue + 6, 200, 255]
    c5_lower = [c5_hue - 6, 70, 100]

    c6_hue = c6_hueOG + cv2.getTrackbarPos("Hue 4", 'Hue Sliders')
    c6_upper = [c6_hue + 8, 210, 255]
    c6_lower = [c6_hue - 8, 155, 80]

    #changes data type to match
    c1_upper = np.array(c1_upper, dtype = "uint8")
    c1_lower = np.array(c1_lower, dtype = "uint8")
    c2_upper = np.array(c2_upper, dtype = "uint8")
    c2_lower = np.array(c2_lower, dtype = "uint8")
    c5_upper = np.array(c5_upper, dtype = "uint8")
    c5_lower = np.array(c5_lower, dtype = "uint8")
    c6_upper = np.array(c6_upper, dtype = "uint8")
    c6_lower = np.array(c6_lower, dtype = "uint8")

    #masks make the colors overlaying in the image possible
    c1_mask = cv2.inRange(frame_hsv, c1_lower, c1_upper)
    c1_mask = cv2.erode(c1_mask, kernel, iterations = 1)
    
    #filtered color allows for the frame to be updated
    c1_filtered = cv2.bitwise_or(frame, frame, mask = c1_mask)
    
    c2_mask = cv2.inRange(frame_hsv, c2_lower, c2_upper)
    c2_mask = cv2.erode(c2_mask, kernel, iterations = 1)
    
    c2_filtered = cv2.bitwise_or(frame, frame, mask = c2_mask)
 
    c5_mask = cv2.inRange(frame_hsv, c5_lower, c5_upper)
    c5_mask = cv2.erode(c5_mask, kernel, iterations = 1)
    
    c5_filtered = cv2.bitwise_or(frame, frame, mask = c5_mask)
    
    c6_mask = cv2.inRange(frame_hsv, c6_lower, c6_upper)
    c6_mask = cv2.erode(c6_mask, kernel, iterations = 1)
    
    c6_filtered = cv2.bitwise_or(frame, frame, mask = c6_mask)

    #finished filtered image
    filtered_img = cv2.bitwise_or(c1_filtered, c2_filtered)
    filtered_img = cv2.bitwise_or(filtered_img,c5_filtered)
    filtered_img = cv2.bitwise_or(filtered_img,c6_filtered)
    
    #use of centroidFinder function
    c1_centroid = centroidFinder(c1_mask, frame)
    c2_centroid = centroidFinder(c2_mask, frame)
    c5_centroid = centroidFinder(c5_mask, frame)
    c6_centroid = centroidFinder(c6_mask, frame)
    
    #colored lines displayed on frame
    
    #line connecting top and bottom of jaw, colored blue, GREEN AND PURPLE CONNECTION
    cv2.line(frame, (c5_centroid[0], c5_centroid[1]), (c6_centroid[0], c6_centroid[1]), (255,0,0), 5)
    #line connecting jaw near ear to bottom of jaw, colored green, ORANGE AND PURPLE CONNECTION
    cv2.line(frame, (c6_centroid[0], c6_centroid[1]), (c1_centroid[0], c1_centroid[1]), (0,255,0), 5)
    #line connecting jaw near ear and top of jaw, colored red, ORANGE AND GREEN CONNECTION
    cv2.line(frame, (c1_centroid[0], c1_centroid[1]), (c5_centroid[0], c5_centroid[1]), (0,0,255), 5)
    
    
    #arrays that connect all the colors, not actually in use in the code but this code acts as a demonstration
    #that four colors connected will add up to 360 degrees regardless of shape,
    #that the code can meet the definition of a quadrilateral shape
    c1ar = [int(c2_centroid[0]-c1_centroid[0]), int(c2_centroid[1]-c1_centroid[1])]
    c2ar = [int(c5_centroid[0]-c2_centroid[0]), int(c5_centroid[1]-c2_centroid[1])]
    c5ar = [int(c6_centroid[0]-c5_centroid[0]), int(c6_centroid[1]-c5_centroid[1])]
    c6ar = [int(c1_centroid[0]-c6_centroid[0]), int(c1_centroid[1]-c6_centroid[1])]
    threehundredsixtydegrees = c1ar+c2ar+c5ar+c6ar
    
    
    bigArC1C5 = [int(c5_centroid[0]-c1_centroid[0]), int(c6_centroid[1]-c1_centroid[1])]
    bigArC1C6 = [int(c6_centroid[0]-c1_centroid[0]), int(c6_centroid[1]-c1_centroid[1])]

    c1c2Angle = angle(c1ar, c2ar)
    c2c5Angle = angle(c2ar, c5ar)
    c5c6Angle = angle(c5ar, c6ar)
    c6c1Angle = angle(c6ar, c1ar)
    AngleArray = [c1c2Angle, c2c5Angle, c5c6Angle, c6c1Angle]
    
    #put image in window
    
    cv2.imshow("WebCam", frame)
    
    keypressed = cv2.waitKey(30)

#start time is used to calculate the time elapsed after 'e' is pressed
start_time = time.time()
while keypressed != ord('w'):
    
    ret, frame = cap.read()

    #create HSV image
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        #allows colors to change based on slider input
    c1_hue = c1_hueOG + cv2.getTrackbarPos("Hue 1", 'Hue Sliders')
    c1_upper = [c1_hue + 3, 255, 255]
    c1_lower = [c1_hue - 3, 160, 160]

    c2_hue = c2_hueOG + cv2.getTrackbarPos("Hue 2", 'Hue Sliders')
    c2_upper = [c2_hue + 4, 255, 215]
    c2_lower = [c2_hue - 4, 140, 55]

    c5_hue = c5_hueOG + cv2.getTrackbarPos("Hue 3", 'Hue Sliders')
    c5_upper = [c5_hue + 6, 200, 255]
    c5_lower = [c5_hue - 6, 70, 100]

    c6_hue = c6_hueOG + cv2.getTrackbarPos("Hue 4", 'Hue Sliders')
    c6_upper = [c6_hue + 8, 210, 255]
    c6_lower = [c6_hue - 8, 155, 80]

    #changes data type to match
    c1_upper = np.array(c1_upper, dtype = "uint8")
    c1_lower = np.array(c1_lower, dtype = "uint8")
    c2_upper = np.array(c2_upper, dtype = "uint8")
    c2_lower = np.array(c2_lower, dtype = "uint8")
    c5_upper = np.array(c5_upper, dtype = "uint8")
    c5_lower = np.array(c5_lower, dtype = "uint8")
    c6_upper = np.array(c6_upper, dtype = "uint8")
    c6_lower = np.array(c6_lower, dtype = "uint8")

    #masks make the colors overlaying in the image possible
    c1_mask = cv2.inRange(frame_hsv, c1_lower, c1_upper)
    c1_mask = cv2.erode(c1_mask, kernel, iterations = 1)
    
    #filtered color allows for the frame to be updated
    c1_filtered = cv2.bitwise_or(frame, frame, mask = c1_mask)
    
    c2_mask = cv2.inRange(frame_hsv, c2_lower, c2_upper)
    c2_mask = cv2.erode(c2_mask, kernel, iterations = 1)
    
    c2_filtered = cv2.bitwise_or(frame, frame, mask = c2_mask)
 
    c5_mask = cv2.inRange(frame_hsv, c5_lower, c5_upper)
    c5_mask = cv2.erode(c5_mask, kernel, iterations = 1)
    
    c5_filtered = cv2.bitwise_or(frame, frame, mask = c5_mask)
    
    c6_mask = cv2.inRange(frame_hsv, c6_lower, c6_upper)
    c6_mask = cv2.erode(c6_mask, kernel, iterations = 1)
    
    c6_filtered = cv2.bitwise_or(frame, frame, mask = c6_mask)

    #finished filtered image
    filtered_img = cv2.bitwise_or(c1_filtered, c2_filtered)
    filtered_img = cv2.bitwise_or(filtered_img,c5_filtered)
    filtered_img = cv2.bitwise_or(filtered_img,c6_filtered)
    
    #use of centroidFinder function
    c1_centroid = centroidFinder(c1_mask, frame)
    c2_centroid = centroidFinder(c2_mask, frame)
    c5_centroid = centroidFinder(c5_mask, frame)
    c6_centroid = centroidFinder(c6_mask, frame)
    
    #colored lines displayed on frame
    
    #line connecting top and bottom of jaw, colored blue, GREEN AND PURPLE CONNECTION
    cv2.line(frame, (c5_centroid[0], c5_centroid[1]), (c6_centroid[0], c6_centroid[1]), (255,0,0), 5)
    #line connecting jaw near ear to bottom of jaw, colored green, ORANGE AND PURPLE CONNECTION
    cv2.line(frame, (c6_centroid[0], c6_centroid[1]), (c1_centroid[0], c1_centroid[1]), (0,255,0), 5)
    #line connecting jaw near ear and top of jaw, colored red, ORANGE AND GREEN CONNECTION
    cv2.line(frame, (c1_centroid[0], c1_centroid[1]), (c5_centroid[0], c5_centroid[1]), (0,0,255), 5)
    
    #arrays that connect all the colors, not actually in use in the code but this code acts as a demonstration
    #that four colors connected will add up to 360 degrees regardless of shape,
    #that the code can meet the definition of a quadrilateral shape
    c1ar = [int(c2_centroid[0]-c1_centroid[0]), int(c2_centroid[1]-c1_centroid[1])]
    c2ar = [int(c5_centroid[0]-c2_centroid[0]), int(c5_centroid[1]-c2_centroid[1])]
    c5ar = [int(c6_centroid[0]-c5_centroid[0]), int(c6_centroid[1]-c5_centroid[1])]
    c6ar = [int(c1_centroid[0]-c6_centroid[0]), int(c1_centroid[1]-c6_centroid[1])]
    threehundredsixtydegrees = c1ar+c2ar+c5ar+c6ar
    
    bigArC5C1 = [int(c5_centroid[0]-c1_centroid[0]), int(c5_centroid[1]-c1_centroid[1])]
    bigArC6C1 = [int(c6_centroid[0]-c1_centroid[0]), int(c6_centroid[1]-c1_centroid[1])]

    bigAng =angle(bigArC5C1, bigArC6C1)
    printAng = str(bigAng)
    colorText = [0, 255, 0]
    if bigAng != None:
        floatAng = float(int(bigAng))
        if floatAng <= 40 or floatAng >=90:
            colorText = [0,0,255]
            cv2.putText(frame, printAng, (100,100), 0, 4, colorText)
        else:
            colorText = [0,255,0]
            cv2.putText(frame, printAng, (100,100), 0, 4, colorText)
    else:
        cv2.putText(frame, "No Angle", (50,50), 0, 4, [0,0,0])
    
    #put image in window
    
    cv2.imshow("WebCam", frame)
        
    angle_yes = bigAng
    end_time = time.time()
    time_true = end_time - start_time
    data_yay = [time_true, angle_yes]
    # write a row to the csv file
    writer.writerow(data_yay)
    
    keypressed = cv2.waitKey(30)
    

cv2.destroyAllWindows()
cap.release()
f.close()

