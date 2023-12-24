import cv2
import numpy as np
import glob
import re
import os
import winsound
import time

def slope_averaging_custom(image, lines):
    """
    Input Parameters:
    image: Grayscale image
    lines: Detected lines from Hough transform

    Functionality:
    Calculates slope and intercept for each line.
    Averages slopes for left and right lanes separately.
    """
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept_point = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept_point))
        else:
            right_fit.append((slope, intercept_point))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = coordinates_maker(image, left_fit_average)
    right_line = coordinates_maker(image, right_fit_average)
    return np.array([left_line, right_line])


def coordinates_maker(image, line_parameters):
    """
    Input Parameters:
    image: Grayscale image
    line_parameters: Tuple typed object of slope and intercept

    Functionality:
    Computes coordinates of a line based on slope and intercept.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def find_edge_pixels(edge_detected_image, center_x_position):
    """
    Input Parameters:
    edge_detected_image: Binary edge-detected image
    center_x_position: X-coordinate of the center

    Functionality:
    Finds left and right edge pixels starting from the center.
    """
    height, width = edge_detected_image.shape
    left_edges = []
    right_edges = []

    for y in range(height//2, height):  # Start from the middle of the image and go to the bottom (TODO - check the explanation related to this)
        # Search to the left from the center
        for x_left in range(center_x_position, -1, -1):
            if edge_detected_image[y, x_left] == 255:
                left_edges.append((x_left, y))
                break

        # Search to the right from the center
        for x_right in range(center_x_position, width):
            if edge_detected_image[y, x_right] == 255:
                right_edges.append((x_right, y))
                break

    return left_edges, right_edges

def sobel_edge_detection(image):
    """
    Input Parameters:
    image: Grayscale image

    Functionality:
    Applies Sobel edge detection to the grayscale image.
    """
    # Ensure the image is in BGR format
    if len(image.shape) == 2:
        # Convert single-channel image to 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        # Convert single-channel image to 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_coordinates = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) #TODO Why kernam size 3 is used
    return sobel_coordinates

def sobel_edge_visualization(image):
    """
    Input Parameters:
    image: Grayscale image

    Functionality:
    Applies Sobel edge detection to the grayscale image.(visualization purposes)
    """
    # Convert the image to float32
    image = image.astype(np.float32)
    blurred_image = cv2.GaussianBlur(image, (9, 9), 0)

    # Apply Sobel operator in both horizontal and vertical directions
    sobel_x_coordinates = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_coordinates = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    gradient_magnitude = cv2.magnitude(sobel_x_coordinates, sobel_y_coordinates)

    # Apply a threshold to the gradient magnitude to create a binary edge-detected image
    threshold_value = 60
    _, edge_detected = cv2.threshold(gradient_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

    return edge_detected.astype(np.uint8)

def find_and_visualize_edge_pixels(image, center_x_position):
    """
    Input Parameters:
    image: Grayscale image
    center_x_position: X-coordinate of the center defined

    Functionality:
    Combines Sobel edge detection and edge pixel visualization functi9onality.
    """
    edge_detected = sobel_edge_visualization(image)
    left_edges, right_edges = find_edge_pixels(edge_detected, center_x_position)

    # Visualization
    img_visualized = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for x, y in left_edges:
        cv2.circle(img_visualized, (x, y), 1, (0, 0, 255), -1)  # Red for left side
    for x, y in right_edges:
        cv2.circle(img_visualized, (x, y), 1, (0, 255, 0), -1)  # Green for right side
    cv2.line(img_visualized, (center_x_position, 0), (center_x_position, img_visualized.shape[0]), (255, 0, 0), 2)  # Blue line

    return img_visualized

def line_intersection_coordinate(lines):
    """
    Input Parameters:
    lines: List of two lines

    Functionality:
    Calculates the intersection point coordinates of two lines.
    """
    x_diff = (lines[0][0] - lines[0][2], lines[1][0] - lines[1][2])
    y_diff = (lines[0][1] - lines[0][3], lines[1][1] - lines[1][3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return 0, 0

    d = (det(*((lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]))),
         det(*((lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]))))
    x = int(det(d, x_diff) / div)
    y = int(det(d, y_diff) / div)

    return x, y


def generate_lines(image, lines):
    """
    Input Parameters:
    image: Grayscale image
    lines: List of lines

    Functionality:
    Draw lines.
    """
    end_x, end_y = line_intersection_coordinate(lines)

    two_lanes = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(two_lanes, (x1, y1), (end_x, end_y), (255, 0, 0), 5)

    cv2.circle(two_lanes, (end_x, end_y), 10, (255, 255, 0), 2)
    return two_lanes


def process_area(image):
    """
    Input Parameters:
    image: Grayscale image

    Functionality:
    Defines a region to consider and creates a mask.
    """
    polygons = np.array([[(0, image.shape[0]), (0, image.shape[0] - 100), (290, 320), (340, 320),
                          (image.shape[1] - 150, image.shape[0])]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(img):
    """
    Input Parameters:
    img: Edge detected image

    Functionality:
    Performs the Hough transform operation to detect lines.
    """
    pixels = np.array(img)
    height, width = pixels.shape
    diag_len = int(np.sqrt(height ** 2 + width ** 2))
    hough_space = np.zeros((2 * diag_len, 180), dtype=np.uint64)
    thetas = np.deg2rad(np.arange(-90, 90))
    for y in range(height):
        for x in range(width):
            if pixels[y, x] > 128:
                for theta_index, theta in enumerate(thetas):
                    rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
                    hough_space[rho, theta_index] += 1

    return hough_space, thetas, diag_len


def find_lines(hough_space, thetas, diag_len, threshold=100):
    """
    Input Parameters:
    hough_space: Accumulator array from Hough transform
    thetas: List of angles
    diag_len: Length of the diagonal
    threshold: Threshold for line detection

    Functionality:
    Extracts lines from Hough space based on a threshold defined.
    """
    peaks = np.where(hough_space > threshold)
    lines = []
    for i in range(len(peaks[0])):
        rho = peaks[0][i] - diag_len
        theta = thetas[peaks[1][i]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.append(np.array([(x1, y1), (x2, y2)]))
    return lines


def detect_lanes(img):
    """
    Input Parameters:
    img: Grayscale image

    Functionality:
    Combines Sobel edge detection, Hough transform, and
    line filtering to detect and visualize lanes.
    """
    sobel = sobel_edge_detection(img)
    cropped_Image = process_area(sobel)
    edges = cv2.convertScaleAbs(cropped_Image, alpha=1, beta=0)
    thresh, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(edges, (3, 3))

    hough_space, thetas, diag_len = hough_transform(edges)
    lines = find_lines(hough_space, thetas, diag_len, threshold=100)

    filtered_lines = slope_averaging_custom(img, lines)
    two_lanes = generate_lines(img, filtered_lines)

    # Visualization
    img_visualized = cv2.addWeighted(img, 0.8, two_lanes, 1, 1)

    return img_visualized


def video_generator(input_folder, output_folder, name):
    """
    Generates a video from a sequence of images.

    Args:
        input_folder: Path to input image folder.
        output_folder: Path to save the output video.
        name: Name of the output video file (without extension).
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    images = []
    filenames = []
    for filename in glob.glob(os.path.join(input_folder, '*.bmp')):
        filenames.append(filename)
        img = cv2.imread(filename)

    filenames.sort(key=natural_keys)

    for file in filenames:
        img = cv2.imread(file)
        height, width, layers = img.shape
        size = (width, height)
        images.append(img)

    out = cv2.VideoWriter(os.path.join(output_folder, f'{name}.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()


def main():
    start_time = time.time()
    video_folders = ['TestVideo_1', 'TestVideo_2']

    for video_folder_name in video_folders:

        # Create output folders if not exists
        output_folder = os.path.join('outputs', video_folder_name)
        os.makedirs(output_folder, exist_ok=True)

        loaded_images_folder = os.path.join(output_folder, '01_Loaded_images')
        gray_scaled_images_folder = os.path.join(output_folder, '02_Grey_Scaled_images')
        sobel_edge_detected_folder = os.path.join(output_folder, '03_Sobel_edge_detected_images')
        sobel_edge_visualization_folder = os.path.join(output_folder, '04_Sobel_edge_viaulization_images')
        hough_transfromed_folder = os.path.join(output_folder, '05_HoughTransformed_images')
        visualized_result_folder = os.path.join(output_folder, '06_Visualized_Result')

        folders = [loaded_images_folder, gray_scaled_images_folder, sobel_edge_detected_folder, sobel_edge_visualization_folder, hough_transfromed_folder,
                   visualized_result_folder]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        # Process images
        image_list = [file for file in os.listdir(video_folder_name) if file.endswith(".bmp")]

        # Assuming the first image in the folder determines the shape of subsequent images
        first_image_path = os.path.join(video_folder_name, image_list[0])
        img = cv2.imread(first_image_path)
        center_x_position = img.shape[1] // 2 - 140 if 'Right' in image_list[0] else img.shape[1] // 2 - 140

        for image_name in image_list:
            image_path = os.path.join(video_folder_name, image_name)

            # Load Image
            img = cv2.imread(image_path)
            cv2.imwrite(os.path.join(loaded_images_folder, f"{image_name[:-4]}_Loaded.bmp"), img)

            # Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(gray_scaled_images_folder, f"{image_name[:-4]}_Grey_Scaled.bmp"), gray)

            # Sobel Edge Detection
            edge_detected = sobel_edge_detection(gray)
            cv2.imwrite(os.path.join(sobel_edge_detected_folder, f"{image_name[:-4]}_Sobel_Edge_Detected.bmp"),
                        edge_detected)

            # Sobel Edge Visualaization
            edge_visualaization = sobel_edge_visualization(gray)

            # Debugging: Print the shape of the edge_detected image
            print(f"Shape of edge_detected image for {image_name}: {edge_visualaization.shape}")

            # Check if the image is empty
            if edge_visualaization.size == 0:
                print(f"Error: edge_detected image is empty for {image_name}")
                continue

            cv2.imwrite(os.path.join(sobel_edge_detected_folder, f"{image_name[:-4]}_Sobel_Edge_Detected.bmp"), edge_detected)

            # Edge Pixel Found and Visualize
            img_visualized = find_and_visualize_edge_pixels(gray, center_x_position)
            cv2.imwrite(os.path.join(sobel_edge_visualization_folder, f"{image_name[:-4]}_Sobel_Edge_Visualization.bmp"),
                        img_visualized)


            # Edge Pixel Found and Visualize
            img_visualized = detect_lanes(img)
            cv2.imwrite(os.path.join(hough_transfromed_folder, f"{image_name[:-4]}_Hough_transformed_image.bmp"), img_visualized)

        # Video output
        video_generator(hough_transfromed_folder, visualized_result_folder, video_folder_name)
    end_time = time.time()

    # Calculate the time taken
    elapsed_time = end_time - start_time

    print(f"Time taken: {elapsed_time:.2f} seconds")

    winsound.Beep(1000, 2000)

if __name__ == "__main__":
    main()
