import cv2
import os
import datetime

video = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")

fps = video.get(cv2.CAP_PROP_FPS)

five_seconds = 5*fps

count = 1

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    elif count % five_seconds == 0:
        cv2.imwrite(os.path.join('images', datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '-' + str(count) + '.jpg'), frame)
        print("Arquivo Salvo com sucesso")
    count = count + 1

video.release()
cv2.destroyAllWindows()