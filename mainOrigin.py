import cv2
import numpy as np
import sys
import mediapipe as mp
from filterpy.kalman import KalmanFilter
""" import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import filedialog, messagebox """


# Função chamada quando o botão "Vídeo" é clicado
""" 
def selectVideo():
    global videoSource
    file_path = filedialog.askopenfilename(filetypes=[("Arquivos de vídeo", "*.mp4")])
    if file_path:
        videoSource = cv2.VideoCapture(file_path)
        videoSource.set(3, realWidth)
        videoSource.set(4, realHeight)
        messagebox.showinfo("Info", f"Modo: Vídeo - Arquivo: {file_path}")
        fps = videoSource.get(cv2.CAP_PROP_FPS)
        # if fps <= 30: fps = 45
        print("FPS:", fps)
        window.destroy()
        startProcessing(57)
        print("Info", f"Modo: Vídeo - Arquivo: {file_path}")
# Função chamada quando o botão "Webcam" é clicado
def selectWebcam():
    global videoSource
    if len(sys.argv) == 2:
        videoSource = cv2.VideoCapture(sys.argv[1])
    else:
        videoSource = cv2.VideoCapture(0)
    videoSource.set(3, realWidth)
    videoSource.set(4, realHeight)
    messagebox.showinfo("Info", "Modo: Webcam")
    fps = videoSource.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    window.destroy()
    startProcessing(fps) """
# Cria a janela da interface gráfica
# window = Tk()

# Cria os botões
""" 
videoButton = Button(window, text="Vídeo", command=selectVideo)
videoButton.pack(side=LEFT, padx=10, pady=10)

webcamButton = Button(window, text="Webcam", command=selectWebcam)
webcamButton.pack(side=LEFT, padx=10, pady=10)
 """

def startProcessing(fps, video):
    realWidth = 640
    realHeight = 480
    videoWidth = 640
    videoHeight = 480
    # initialize BPM measurements per frame 
    bpmmeasurements = np.empty((0, 2))

    # Add Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    videoSource = cv2.VideoCapture(video)

    # Helper Methods
    def buildGauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

        
    realWidth = 640
    realHeight = 480
    videoWidth = 640
    videoHeight = 480

    videoChannels = 3
    videoFrameRate = fps


    # Output Videos
    if len(sys.argv) != 2:
        originalVideoFilename = "original.avi"
        originalVideoWriter = cv2.VideoWriter()
        originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

    outputVideoFilename = "BPM.avi"
    outputVideoWriter = cv2.VideoWriter()
    outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

    # Duração média do período cardíaco em segundos
    duracao_periodo_cardiaco = 1.5

    # Taxa de quadros do vídeo
    taxa_quadros = videoFrameRate

    # Margem de segurança
    margem_seguranca = 10

    # Cálculo do buffer size
    bufferSize = int(duracao_periodo_cardiaco * taxa_quadros) + margem_seguranca
    # Color Magnification Parameters
    levels = 3
    alpha = 170
    minFrequency = 0.5
    maxFrequency = 3.0
    bufferIndex = 0

    # Output Display Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (20, 30)
    bpmTextLocation = (videoWidth//2 + 5, 30)
    fontScale = 0.5  # Modify the font scale to make the text half the size
    fontColor = (0, 255, 0)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 3

    # Initialize Gaussian Pyramid
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels+1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    # Bandpass Filter for Specified Frequencies
    frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    # Heart Rate Calculation Variables
    bpmCalculationFrequency = 15
    bpmBufferIndex = 0
    bpmBufferSize = 150
    bpmBuffer = np.zeros((bpmBufferSize))

    kalman_filter = KalmanFilter(dim_x=1, dim_z=1)
    initial_bpm_estimate = 60
    initial_covariance = 1000

    kalman_filter.x = np.array([[initial_bpm_estimate]])
    kalman_filter.F = np.array([1])  # state transition matrix
    kalman_filter.H = np.array([1])  # measurement function
    kalman_filter.P *= initial_covariance  # covariance matrix
    kalman_filter.Q = 1e-4  # process noise
    kalman_filter.R = np.array([[1]])  # measurement noise
    i = 0  # Initialize i for BPM calculation

    bpm_values = []

    continuar = True
    while (continuar):
        ret, frame = videoSource.read()
        
        if ret == False:
            break

        if len(sys.argv) != 2:
            originalFrame = frame.copy()
            originalVideoWriter.write(originalFrame)

        # Use MediaPipe Face Detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)

        # Check if faces are detected
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Ignore if bounding box is too close to the edge
                if bbox[0] < 10 or bbox[1] < 10 or bbox[0] + bbox[2] > iw - 10 or bbox[1] + bbox[3] > ih - 10:
                    continue
                    
                detectionFrame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
                detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight)) # resize detectionFrame

                # Construct Gaussian Pyramid
                videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
                fourierTransform = np.fft.fft(videoGauss, axis=0)

                # Bandpass Filter
                fourierTransform[mask == False] = 0

                # Grab a Pulse
                if bufferIndex % bpmCalculationFrequency == 0:
                    i = i + 1
                    for buf in range(bufferSize):
                        fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                    hz = frequencies[np.argmax(fourierTransformAvg)]
                    bpm = 60.0 * hz
                    bpmBuffer[bpmBufferIndex] = bpm
                    bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
                

                # Amplify
                filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
                filtered = filtered * alpha

                # Reconstruct Resulting Frame
                filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
                outputFrame = detectionFrame + filteredFrame
                outputFrame = cv2.convertScaleAbs(outputFrame)

                bufferIndex = (bufferIndex + 1) % bufferSize

                frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = cv2.resize(outputFrame, (bbox[2], bbox[3])) # resize outputFrame to fit original bbox size
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), boxColor, boxWeight)


                # Grab a Pulse
                if bufferIndex % bpmCalculationFrequency == 0:
                    i = i + 1
                    for buf in range(bufferSize):
                        fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                    hz = frequencies[np.argmax(fourierTransformAvg)]
                    bpm = 60.0 * hz

                    # Kalman Filter Update
                    kalman_filter.predict()
                    kalman_filter.update(np.array([[bpm]]))
                    filtered_bpm = kalman_filter.x.item()

                    bpmBuffer[bpmBufferIndex] = filtered_bpm
                    bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

                    bpm_values.append(filtered_bpm)

                if i >= 30 * videoFrameRate:  # 30 segundos
                    filtered_bpm_values = [bpm for bpm in bpm_values if bpm > 60]
                    print(filtered_bpm_values)
                    if filtered_bpm_values:
                        bpm_average = np.mean(filtered_bpm_values)
                        bpm_mode = np.argmax(np.bincount(filtered_bpm_values))
                        bpm_median = np.median(filtered_bpm_values)
                        print("BPM Médio:", bpm_average)
                        print("BPM Moda:", bpm_mode)
                        print("BPM Mediana:", bpm_median)
                        continuar = False
                    else:
                        print("Nenhum valor de BPM acima de 60 encontrado.")
                    bpm_values = []
                    i = 0

                i += 1

                # Display BPM
                if i > bpmBufferSize:
                    # Update BPM Array with the latest BPM and frame number
                    bpmmeasurements = np.append(bpmmeasurements, [[i, bpmBuffer.mean()]], axis=0)

                    # Calculate the average BPM and the lowest and highest BPM
                    averageBPM = bpmmeasurements[:, 1].mean()
                    lowestBPM = bpmmeasurements[:, 1].min()
                    highestBPM = bpmmeasurements[:, 1].max()

                    # Display the average BPM, lowest BPM, and highest BPM
                    cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), (20, 60), font, 0.5, fontColor, lineType)
                    cv2.putText(frame, "Media BPM: %d" % averageBPM, (20, 90), font, 0.5, fontColor, lineType)
                    cv2.putText(frame, "Mais Baixo BPM: %d" % lowestBPM, (20, 120), font, 0.5, fontColor, lineType)
                    cv2.putText(frame, "Mais Alto BPM: %d" % highestBPM, (20, 150), font, 0.5, fontColor, lineType)
            
                    
                
                else:
                    cv2.putText(frame, "Calculating BPM...", (20, 30), font, 0.5, fontColor, lineType)

                outputVideoWriter.write(frame)

                if len(sys.argv) != 2:
                    cv2.imshow("Webcam Heart Rate Monitor", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        continuar = False
                        print('Processo Cancelado!')
   
    videoSource.release()
    cv2.destroyAllWindows()
    outputVideoWriter.release()
    if len(sys.argv) != 2:
        originalVideoWriter.release() 
# window.mainloop()









