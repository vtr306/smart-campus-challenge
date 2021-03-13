import cv2
import time
import os
import datetime

video = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    cv2.imwrite(os.path.join('images', datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '.jpg'), frame)
    print("Arquivo Salvo com sucesso")
    time.sleep(5)

video.release()
cv2.destroyAllWindows()
