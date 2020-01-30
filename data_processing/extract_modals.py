import subprocess
import os
import cv2
import numpy as np
import os
from tqdm import tqdm
from data_processing.paths import data_path, audio_path, image_path


if not os.path.isdir(audio_path):
    os.mkdir(audio_path)

if not os.path.isdir(image_path):
    os.mkdir(image_path)

for file in tqdm(os.listdir(data_path)):
    #extract audio from file

    command = f"ffmpeg -f -i {os.path.join(data_path, file)} -ab 160k -ac 2 -ar 44100 -vn {os.path.join(audio_path, file[:-4])}.wav"
    subprocess.run(command, shell=True)

    # extract video frame

    cap = cv2.VideoCapture(os.path.join(data_path, file))
    if os.path.isdir(os.path.join(image_path, file[:-4])) == False:
        currentFrame = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            # Saves image of the current frame in jpg file
            if not os.path.isdir(os.path.join(image_path, file[:-4])):
                os.mkdir(os.path.join(image_path, file[:-4]))
            name = os.path.join(image_path, file[:-4], f'frame_{str(currentFrame)}.jpg')
            # print('Creating... ' + name)
            cv2.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()