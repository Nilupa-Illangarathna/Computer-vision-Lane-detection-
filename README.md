# Lane Detection Project

Lane Detection Project is a Python script designed for detecting and visualizing lane lines in a sequence of images extracted from a video feed. The script employs edge detection techniques, Hough Transform, and line averaging to identify and display lane lines on the road.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The project focuses on automating the process of lane detection for video data. It incorporates image processing techniques, including Sobel edge detection and the Hough Transform, to identify potential lane lines. The script processes each frame of the video, applies filtering and averaging to obtain the left and right lane lines, and visualizes the result.

## Key Features
- **Edge Detection:** Utilizes Sobel edge detection to enhance edges in the input images.
- **Hough Transform:** Applies the Hough Transform to identify lines in polar coordinates.
- **Lane Averaging:** Averages the slopes and intercepts of the left and right lane lines for robust detection.
- **Visualization:** Generates visualizations of detected lanes and an output video illustrating the lane detection process.

## Dependencies
- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation
1. Clone the repository:
   ```bash
   [git clone https://github.com/your-username/lane-detection-project.git
   cd lane-detection-project](https://github.com/Nilupa-Illangarathna/Computer-vision-Lane-detection-.git)https://github.com/Nilupa-Illangarathna/Computer-vision-Lane-detection-.git

## Methodology

<table>
  <tr>
    <td> </td>
     <td> </td>
     <td> </td>
  </tr>
  <tr>
    <td><img src="https://github.com/Nilupa-Illangarathna/Computer-vision-Lane-detection-/assets/95247831/e214df76-6bde-4ed1-af86-b3b89fc299f1" width=300></td>
    <td><img src="https://github.com/Nilupa-Illangarathna/Computer-vision-Lane-detection-/assets/95247831/53e770a0-04e8-4ea9-a6c0-3a654a43ef92" width=300></td>
    <td><img src="https://github.com/Nilupa-Illangarathna/Computer-vision-Lane-detection-/assets/95247831/2cfd4412-a55f-4927-85f9-55632d4b51fc" width=300></td>
  </tr>
 </table>

## Folder Structure
- **outputs/**
  - **TestVideo_1/**
    - **01_Loaded_images/**
      - *Example_Image_1_Loaded.bmp*
      - *Example_Image_2_Loaded.bmp*
      - ...
    - **02_Grey_Scaled_images/**
      - *Example_Image_1_Grey_Scaled.bmp*
      - *Example_Image_2_Grey_Scaled.bmp*
      - ...
    - **03_Sobel_edge_detected_images/**
      - *Example_Image_1_Sobel_Edge_Detected.bmp*
      - *Example_Image_2_Sobel_Edge_Detected.bmp*
      - ...
    - **04_EdgePixelFound_images/**
      - *Example_Image_1_Edge_Pixel_Found.bmp*
      - *Example_Image_2_Edge_Pixel_Found.bmp*
      - ...
    - **05_Visualized_Result/**
      - *Example_Image_1_Visualized_Result.bmp*
      - *Example_Image_2_Visualized_Result.bmp*
      - ...
  - **TestVideo_2/**
    - *(Similar structure as TestVideo_1)*
  - ...
- **input/**
  - **TestVideo_1/**
   - *frame_0000.bmp*
   - *frame_0001.bmp*

## Results
Visualized images and output videos are saved in the respective folders within the `outputs` directory. The folder structure is organized to provide a step-by-step view of the image processing pipeline, from loading and preprocessing to final lane visualization.

