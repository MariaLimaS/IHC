# Importa as bibliotecas necessárias
import cv2  # OpenCV para processamento de imagem e detecção de faces
import numpy as np  # NumPy para manipulação de matrizes
from keras.preprocessing.image import img_to_array  # Função para converter imagem em matriz
from tensorflow.keras.models import load_model  # Carrega modelos de aprendizado profundo

# Lista das expressões faciais que o modelo pode detectar
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

# Caminhos para os arquivos do modelo Haar Cascade para detecção de faces e o modelo treinado
cascade_faces = 'haarcascade_frontalface_default.xml'
caminho_modelo = 'modelo_01_expressoes.h5'

# Inicializa o modelo Haar Cascade para detecção de faces
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_faces)

# Carrega o modelo de reconhecimento de expressões faciais
classificador_emocoes = load_model(caminho_modelo)

# Compila o modelo de reconhecimento de expressões
classificador_emocoes.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Inicializa a captura de vídeo da webcam (0 representa a webcam padrão)
webcam = cv2.VideoCapture(0)

# Verifica se a captura foi aberta com sucesso
if not webcam.isOpened():
    print("Erro ao abrir a webcam.")
else:
    while True:
        # Captura o próximo quadro da webcam
        validacao, frame = webcam.read()

        # Verifica se a leitura do frame foi bem-sucedida
        if not validacao:
            print("Erro ao ler o frame.")
            break

        original = frame  # Faz uma cópia do quadro capturado

        # Detecta faces no quadro usando o modelo Haar Cascade
        faces = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        # Converte o quadro para tons de cinza para processamento mais eficiente
        cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Se uma ou mais faces forem detectadas
        if len(faces) > 0:
            for (fX, fY, fW, fH) in faces:
                # Extrai a Região de Interesse (ROI) da face
                roi = cinza[fY:fY + fH, fX:fX + fW]

                # Redimensiona a ROI para 48x48 pixels (formato esperado pelo modelo)
                roi = cv2.resize(roi, (48, 48))

                # Normaliza os valores dos pixels para o intervalo [0, 1]
                roi = roi.astype("float") / 255.0

                # Converte a ROI em uma matriz NumPy
                roi = img_to_array(roi)

                # Expande as dimensões da matriz para que possa ser usada como entrada no modelo
                roi = np.expand_dims(roi, axis=0)

                # Faz a previsão das expressões faciais para a ROI usando o modelo
                preds = classificador_emocoes.predict(roi)[0]

                # Calcula a emoção mais provável com base nas previsões do modelo
                emotion_probability = np.max(preds)
                label = expressoes[preds.argmax()]

                # Desenha um retângulo ao redor da face detectada para destacá-la
                cv2.rectangle(original, (fX, fY), (fX + fW, fY + fH), (255, 0, 255), 2)

                # Mostra um gráfico de barras das emoções se apenas uma face for detectada
                if len(faces) == 1:
                    for (i, (emotion, prob)) in enumerate(zip(expressoes, preds)):
                        # Nome das emoções
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        w = int(prob * 300)
                        cv2.rectangle(original, (7, (i * 35) + 5), (w, (i * 35) + 35), (255, 0, 255), -1)
                        cv2.putText(original, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            print('Nenhuma face detectada')

        # Exibe o quadro da webcam com as informações
        cv2.imshow("Camera", original)

        # Encerra o loop quando a tecla 'Esc' é pressionada
        if cv2.waitKey(5) == 27:
            break

    # Libera a captura de vídeo e fecha a janela
    webcam.release()
    cv2.destroyAllWindows()
