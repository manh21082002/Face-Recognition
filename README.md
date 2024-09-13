\documentclass{article}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Face Recognition Project - README}
\author{}
\date{}

\begin{document}

\maketitle

\section{Overview}

This project demonstrates a facial recognition system built with \textbf{OpenCV}, \textbf{Python}, and \textbf{Deep Learning} techniques. The core library used is \texttt{face\_recognition}, which simplifies the process of recognizing faces through pre-trained models from \texttt{dlib} that generate \textbf{128-dimensional face embeddings}.

\section{Libraries Required}

To run this project, install the following libraries:

\begin{verbatim}
pip install dlib
pip install face_recognition
\end{verbatim}

\section{Project Structure}

The project contains the following folders and files:

\begin{itemize}
    \item \texttt{dataset/}: Stores images of individuals, organized into subdirectories (one for each person).
    \item \texttt{test\_images/}: Contains images for testing that are not part of the dataset.
    \item \texttt{output/}: Stores processed video files that have gone through face recognition.
    \item \texttt{videos/}: Stores input video files for face recognition.
\end{itemize}

Other important files include:

\begin{itemize}
    \item \texttt{build\_dataset.py}: Script to capture and build the dataset using a webcam.
    \item \texttt{encode\_faces.py}: Script to encode face images into 128-dimensional vectors (face embeddings).
    \item \texttt{recognizer\_faces\_image.py}: Recognizes faces from images using the encodings generated from the dataset.
    \item \texttt{recognizer\_faces\_video.py}: Recognizes faces from video input (supports both live webcam feed and pre-recorded video).
    \item \texttt{encoding.pickle}: Stores the generated encodings from the dataset for later use in recognition.
\end{itemize}

\section{Steps}

\subsection{Step 1: Build the Dataset}
Run \texttt{build\_dataset.py} to capture images of individuals via webcam. Each person’s images are stored in their respective subdirectory inside the \texttt{dataset/} folder. Ensure to capture at least \textbf{10-20 images} per individual with varying poses and lighting conditions for better accuracy.

\textbf{Note}: Each image should contain only one person’s face to avoid complications in recognition.

\subsection{Step 2: Generate Face Encodings}
After building the dataset, run \texttt{encode\_faces.py} to extract \textbf{face ROIs} (regions of interest) and generate \textbf{128-dimensional embeddings} using pre-trained deep learning models from \texttt{dlib}. These embeddings, along with the corresponding person’s name, are stored in the \texttt{encoding.pickle} file for future use.

\subsection{Step 3: Recognize Faces in Images}
To recognize faces in test images, run \texttt{recognizer\_faces\_image.py}. The script compares the face embeddings of the test image against those in the dataset using \texttt{face\_recognition.compare\_faces()}. The output image will show the recognized faces with their corresponding names.

\subsection{Step 4: Recognize Faces in Videos}
To recognize faces in video streams, run \texttt{recognizer\_faces\_video.py}. This script processes the video frame by frame and compares faces in real-time. For devices with limited computational power (e.g., \textbf{Raspberry Pi}), it's recommended to use the \texttt{hog} method for face detection instead of \texttt{cnn} to balance between speed and accuracy.

\section{Conclusion}

This project provides a complete pipeline for \textbf{face recognition} in both images and video streams using OpenCV and pre-trained models. By extending the project with real-time face detection, it can be adapted for various applications like \textbf{attendance systems} or \textbf{security monitoring}. Further optimizations can be done by implementing additional features like \textbf{fake face detection} or enhancing performance for low-power devices.

\section{References}

\begin{enumerate}
    \item \url{https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213}
    \item \url{https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/}
\end{enumerate}

\end{document}
