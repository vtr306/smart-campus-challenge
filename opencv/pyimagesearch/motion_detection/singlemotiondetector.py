import numpy as np
import imutils
import cv2


class SingleMotionDetector:
    def __init__(self, accumWeight=0.5):
        # armazena o fator accumWeight
        self.accumWeight = accumWeight
        # inicializa o modelo de background
        self.bg = None

    def update(self, image):
        # inicializa o modelo de backgroun se for nulo
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        # atualiza o modelo de background a partir da média ponderada entre o frame,
        # o background e o fator de acumulação.
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        # computa a diferença absoluta entre os pixels entre o frame atual e o
        # modelo de background e se essa diferença for superior ao limite tVal
        # esse pixel se torna branco e é considerado objeto, caso contrário se
        # torna preto e é considerado background
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
        # realiza uma série de erosões e dilatações para tratar a imagem e
        # evitar falsos positivos
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # encontra os contornos na imagem tratada e inicializa as variáveis
        # responsáveis pela formação da área delimitada que informa onde foi
        # encontrado movimento
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        # se nenhum contorno foi encontrado, a imagem é ignorada
        if len(cnts) == 0:
            return None
        # caso contrário, os contornos entram em laço
        for c in cnts:
            # computa a área delimitada de cada contorno e utiliza dela
            # para atualizar os limites da mesma
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        # retorna a localização da área delimitadora para a função
        # primária
        return (thresh, (minX, minY, maxX, maxY))
