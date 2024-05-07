import cv2
import numpy as np
# the above are for the modules which use open cv computer vision.

import getpass
import openpyxl
import datetime
# this is for making the user authentication system

users = {}
# used to store credentials
# with the below code the file containing the passwords are read and stored into user.
with open('user_credentials.txt', 'r') as file:
    for line in file:
        username, password = line.strip().split()
        users[username] = password

user_log_file = "user_log.xlsx"
try:
    user_log_workbook = openpyxl.load_workbook(user_log_file)
except FileNotFoundError:
    user_log_workbook = openpyxl.Workbook()
    user_log_worksheet = user_log_workbook.active
    user_log_worksheet.append(["Username", "Login Time"])

user_log_worksheet = user_log_workbook.active
# Will look for exsisting excel file or will create a new one.

prev_left_line = None
prev_right_line = None


def make_coordinates(image, line_parameters):
    """
    Calculate line coordinates given slope and intercept.

    Args:
        image (numpy.ndarray): The input image.
        line_parameters (tuple): A tuple containing the slope and intercept of the line.

    Returns:
        numpy.ndarray: An array with the coordinates [x1, y1, x2, y2] of the line.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = (y1 - intercept)/slope
    x2 = (y2 - intercept)/slope
    return np.array([x1, y1, x2, y2])


#the function processes the lines detected in the coordinates based on slope
def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    else:
        left_line = None

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    else:
        right_line = None

    if left_line is not None and right_line is not None:
        averaged_lines = np.array([left_line, right_line])
    elif left_line is not None:
        averaged_lines = np.array([left_line])
    elif right_line is not None:
        averaged_lines = np.array([right_line])
    else:
        averaged_lines = None

    return averaged_lines

def canny(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to the input image.

    Args:
        image (numpy.ndarray): The input image.
        low_threshold (int): Lower threshold for Canny edge detection.
        high_threshold (int): Upper threshold for Canny edge detection.

    Returns:
        numpy.ndarray: The edge-detected image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    """
    The above statement makes the image change into grayscale.
    """

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    """
    The statement applies gaussian blub on the gray scale image.
    (5,5) is the kernal of the applied blur.
    0 is the deviation
    """

    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
    scaled_gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    binary_output = np.zeros_like(scaled_gradmag)
    binary_output[(scaled_gradmag >= mag_thresh[0]) & (scaled_gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad_direction = np.arctan2(sobel_y, sobel_x)
    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
    return binary_output

def color_thresh(image, low_thresh=(0, 0, 0), high_thresh=(255, 255, 255)):
    binary_output = np.zeros_like(image[:, :, 0])
    binary_output[(image[:, :, 0] >= low_thresh[0]) & (image[:, :, 0] <= high_thresh[0]) &
                  (image[:, :, 1] >= low_thresh[1]) & (image[:, :, 1] <= high_thresh[1]) &
                  (image[:, :, 2] >= low_thresh[2]) & (image[:, :, 2] <= high_thresh[2])] = 1
    return binary_output

def display_lines(image, lines):
    """
    Create an image with detected lines drawn on it.

    Args:
        image (numpy.ndarray): The input image.
        lines (numpy.ndarray): An array of lines to be drawn.

    Returns:
        numpy.ndarray: An image with lines drawn on it.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.astype(int)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    """
    Apply a region of interest mask to the input image.

    image (numpy.ndarray): The input image.

    numpy.ndarray: The image with the region of interest masked.
    """
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def log_user_login(username):
    # Get the current timestamp
    login_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log the user login time in the Excel file
    user_log_worksheet.append([username, login_time])
    user_log_workbook.save(user_log_file)


def main(video_path):


    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            continue

            sobel_x_binary = abs_sobel_thresh(frame, orient='x', sobel_kernel=3, thresh=(20, 100))
            mag_binary = mag_thresh(frame, sobel_kernel=3, mag_thresh=(30, 100))
            dir_binary = dir_thresh(frame, sobel_kernel=15, thresh=(0.7, 1.3))
            color_binary = color_thresh(frame, low_thresh=(0, 0, 0), high_thresh=(100, 100, 100))


            combined_binary = np.zeros_like(sobel_x_binary)
            combined_binary[((sobel_x_binary == 1) & (mag_binary == 1)) | ((dir_binary == 1) & (color_binary == 1))] = 1


        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()

            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get user credentials and validate login
    while True:
        username = input("Enter your username: ")
        password = getpass.getpass("Enter your password: ")

        if username in users and users[username] == password:
            print("Login successful!")
            log_user_login(username)
            break
        else:
            print("Invalid username or password. Try again.")

    video_path = "test2.mp4"
    main(video_path)

# Save and close the Excel file when the script finishes
user_log_workbook.save(user_log_file)
user_log_workbook.close()
