from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# inicializa o frame que vai ser enviado ao cliente e utiliza do lock
# para garantir o funcionamento do frame impedindo a leitura dele enquanto
# é atualizado
outputFrame = None
lock = threading.Lock()

# inicializa o objeto do framework flask
app = Flask(__name__)

# inicializa a stream do vídeo a partir do link rtsp

vs = VideoStream(src="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov").start()

@app.route("/")
def index():
    # retorna o template renderizado
    return render_template("index.html")


def detect_motion(frameCount):
    # traz variáveis globais para dentro da função
    global vs, outputFrame, lock
    # inicializa o detector de movimento e o número total
    # de frames lidos até o momento
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    # laço que acessa os frames vindos da stream
    while True:
        # acessa o próximo frame da stream, redimensiona a imagem,
        # converte para uma escala de cinza e aplica um filtro para
        # redução de ruído
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # Aplica a data atual no frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # garante que o número mínimo de frames foram lidos para a construção
        # do modelo de background
        if total > frameCount:
            # Detecta se houve movimento no frame
            motion = md.detect(gray)
            # checagem se para confirmar se houve movimento no frame
            # e se houve realizar as ações
            if motion is not None:
                # Desenha uma borda em volta da área de movimento no frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)

        # atualiza o modelo de background e incrementa o número de frames
        # lidos até o momento
        md.update(gray)
        total += 1

        # utilizando do lock, define o frame que vai ser enviado ao cliente
        with lock:
            outputFrame = frame.copy()


def generate():
    # traz variáveis globais para a função
    global outputFrame, lock
    # laço sobre os frames do outputframe
    while True:
        # utilizando-se do lock
        with lock:
            # checa se o frame está disponível
            if outputFrame is None:
                continue
            # codifica o frame em uma imagem jpeg
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # garante que a imagem foi codificada com sucessa
            if not flag:
                continue
        # retorna o outputframe em formato de byte
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # returna a reposta da função generate com um tipo de mídia específico
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # Estrutura de parâmetros necessários para inicialização
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='0.0.0.0',
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=8000,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # inicializa uma thread responsável pela detecção de movimento
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()
    # Inicializa o aplicativo Flask
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

vs.stop()
