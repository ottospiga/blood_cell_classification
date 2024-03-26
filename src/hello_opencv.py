import cv2

import numpy as np
import os
from PIL import Image
import keras
from keras.models import load_model
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import load_model

def main1():
    file = '../datasets/dataset_normal/eosinophil/EO_27.jpg'

    model = load_model('../notebooks/first_test/Model.h5')

    # image = Image.open(file)
    # image = image.resize((80, 80))
    # image = numpy.array(image, dtype = 'float32')
    # image/=255
    # image = image.reshape(1, 80, 80, 3)
    # prediction = model.predict(image)
    # # print(prediction)
    # x = numpy.argmax(prediction)

    # mascara=x
    # # # Create a black image
    # # image = 255 * 0 * (cv2.imread('../datasets/dataset_normal/eosinophil/EO_27.jpg')) 
    # image =  (cv2.imread('../datasets/dataset_normal/eosinophil/EO_27.jpg')) 
    # # print(type(cv2.imread('')))
    # '''
    #     # Write "Hello, World!" on the image
    #     cv2.putText(image, "Hello, World!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #     # Display the image
    #     cv2.imshow("Hello World", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows() # Wait for a keystroke in the window
    # '''
    # # --------------------
    # # Supondo que 'mascara' é a máscara binária da célula
    # # contornos, iasa = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contornos, iasa = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Desenha contornos ao redor de cada célula na imagem original
    # imagem_com_contornos = image.copy()
    # cv2.drawContours(imagem_com_contornos, contornos, -1, (0, 255, 0), 2)

    # # Exibir a imagem resultante
    # cv2.imshow("Células Circundadas", imagem_com_contornos)
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()


    image = Image.open(file)
    image = image.resize((80, 80))
    image = np.array(image, dtype='float32')
    image /= 255
    image = image.reshape(1, 80, 80, 3)
    prediction = model.predict(image)
    x = np.argmax(prediction)
    print(x)
    # Carregar a imagem original
    image_original = cv2.imread(file)

    # Converter a imagem para escala de cinza
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    # Thresholding para binarizar a imagem (convertendo para preto e branco)
    _, thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)

    # Encontrar contornos na imagem binarizada
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha contornos ao redor de cada célula na imagem original
    imagem_com_contornos = image_original.copy()
    cv2.drawContours(imagem_com_contornos, contornos, -1, (0, 255, 0), 2)

    # Exibir a imagem resultante
    cv2.imshow("Células Circundadas", imagem_com_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # file = '../datasets/dataset_normal/eosinophil/EO_27.jpg'
    file = './hemato.jpeg'

    model = load_model('../notebooks/first_test/Model.h5')

    image = Image.open(file)
    image = image.resize((80, 80))
    image = np.array(image, dtype='float32')
    image /= 255
    image = image.reshape(1, 80, 80, 3)
    prediction = model.predict(image)
    x = np.argmax(prediction)
    print(x)

    image_original = cv2.imread(file)

    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    # Thresholding para binarizar a imagem (convertendo para preto e branco)
    _, thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha contornos ao redor de cada célula na imagem original
    imagem_com_contornos = image_original.copy()
    for contorno in contornos:
        # Calcula o retângulo delimitador para cada contorno
        x, y, w, h = cv2.boundingRect(contorno)
        # Desenha o retângulo delimitador ao redor da célula
        cv2.rectangle(imagem_com_contornos, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibir a imagem resultante
    cv2.imshow("Células Circundadas", imagem_com_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

