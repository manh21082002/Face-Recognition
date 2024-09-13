# Face Recognition Project - README

## Overview

This project demonstrates a facial recognition system built with **OpenCV**, **Python**, and **Deep Learning** techniques. The core library used is `face_recognition`, which simplifies the process of recognizing faces through pre-trained models from `dlib` that generate **128-dimensional face embeddings**.

## Libraries Required

To run this project, install the following libraries:

```bash
pip install dlib
pip install face_recognition
```
## Project Structure

The project contains the following folders and files:

- `dataset/`: Stores images of individuals, organized into subdirectories (one for each person).
- `test_images/`: Contains images for testing that are not part of the dataset.
- `output/`: Stores processed video files that have gone through face recognition.
- `videos/`: Stores input video files for face recognition.

Other important files include:

- `build_dataset.py`: Script to capture and build the dataset using a webcam.
- `encode_faces.py`: Script to encode face images into 128-dimensional vectors (face embeddings).
- `recognizer_faces_image.py`: Recognizes faces from images using the encodings generated from the dataset.
- `recognizer_faces_video.py`: Recognizes faces from video input (supports both live webcam feed and pre-recorded video).
- `encoding.pickle`: Stores the generated encodings from the dataset for later use in recognition.

## Steps

### Step 1: Build the Dataset
Run `build_dataset.py` to capture images of individuals via webcam. Each person’s images are stored in their respective subdirectory inside the `dataset/` folder. Ensure to capture at least **10-20 images** per individual with varying poses and lighting conditions for better accuracy.

**Note**: Each image should contain only one person’s face to avoid complications in recognition.

### Step 2: Generate Face Encodings
After building the dataset, run `encode_faces.py` to extract face ROIs (regions of interest) and generate **128-dimensional embeddings** using pre-trained deep learning models from `dlib`. These embeddings, along with the corresponding person’s name, are stored in the `encoding.pickle` file for future use.

### Step 3: Recognize Faces in Images
To recognize faces in test images, run `recognizer_faces_image.py`. The script compares the face embeddings of the test image against those in the dataset using `face_recognition.compare_faces()`. The output image will show the recognized faces with their corresponding names.

### Step 4: Recognize Faces in Videos
To recognize faces in video streams, run `recognizer_faces_video.py`. This script processes the video frame by frame and compares faces in real-time. For devices with limited computational power (e.g., **Raspberry Pi**), it's recommended to use the `hog` method for face detection instead of `cnn` to balance between speed and accuracy.

## Conclusion
This project provides a complete pipeline for **face recognition** in both images and video streams using OpenCV and pre-trained models. By extending the project with real-time face detection, it can be adapted for various applications like **attendance systems** or **security monitoring**. Further optimizations can be done by implementing additional features like **fake face detection** or enhancing performance for low-power devices.

## References
- [https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213](https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213)
- [https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
