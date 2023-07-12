import cv2
import time

def TirarFoto():
    captura = cv2.VideoCapture(0) # Seleciona a Webcam

    face_casacade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Pega o arquivo de detecção de face

    continuar = True # Fala quando o arquivo vai parar

    # Contadores
    i = 50
    i2 = 6

    foto = False # Ainda nenhuma foto foi tirada
    while (continuar):
        # Lê o frame atual da webcam e separa o que vai salvar e o que vai exibir na tela
        ret, frame = captura.read() 
        ret, image = captura.read()

        """ # Inverte a Imagem
        cv2.flip(image, 1)
        cv2.flip(frame, 1) """

        height, width = frame.shape[:2] # Detecta o tamanho da imagem
        
        if not ret: # Verificador se é possivel
            break

        # Capta o centro da webcam
        center_x = int(width / 2)
        center_y = int(height / 2)

        # Tamanho da elipse
        radius_x = 5 * 20
        radius_y = 7 * 20

        # Cor da elipse
        colora = (0,0,255) # Vermelho
        colorb = (255, 0, 0) # Azul
        colorc = (0,255,0) # Verde

        # Desenhar a elipse no centro da tela
        
        
        # Transforma a imagem em escala de cinza e detecta o Rosto
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_casacade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 1:# Verifica se existe apenas 1 rosto na imagem
            
            # Pega as coordenadas do rosto selecionado
            (x, y, w, h) = faces[0]
            face_x = int(x + w / 2)
            face_y = int(y + h / 2)

            # Define uma folga do centro
            folga_xPos = center_x + 50
            folga_xNeg = center_x - 50
            folga_yPos = center_y + 50
            folga_yNeg = center_y - 50

            if face_x <= folga_xPos and face_x >= folga_xNeg and face_y <= folga_yPos and face_y >= folga_yNeg: # rosto dentro da elipse
                cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colorb, thickness=2) # Define a elipse de posicionamento como Azul == Tirando foto
                if i % 10 == 0: 
                    i2 -= 1
                if i2 != 0:
                    cv2.putText(frame, f'Tirando Foto em {i2}',(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorb, thickness=2)
                    
                # Tira a foto quando o contador zerar
                if i <= 0:

                    # Retornos ao usuario
                    cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colorc, thickness=2) 
                    cv2.putText(frame, f'Foto Tirada!',(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorc, thickness=2)
                    foto = True # Fala que a foto foi feita
                i -= 1 # Diminui o contador
            else:
                i = 50
                i2 = 6
                cv2.putText(frame, f'Posicione o rosto no circulo',(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colora, thickness=2)
                cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colora, thickness=2) # Define a elipse de posicionamento como Azul == Tirando foto
        elif len(faces) > 1: # Se tiver mais que um rosto no video
            cv2.putText(frame, f'Mais de um rosto identificado',(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colora, thickness=2)
            cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colora, thickness=2) # Define a elipse de posicionamento como Vermelho == Não vai tirar foto
        else:
            cv2.putText(frame, f'Nenhum rosto detectado',(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colora, thickness=2)
            cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colora, thickness=2) # Define a elipse de posicionamento 
                        

        
        '''
        if len(faces) > 0: # Verifica se existe algum rosto na camera
            
            num_face = len(faces) # Quantidade de rostos detectados
            
            
            indice_face = 0 # Qual é o rosto atual sendo verificado
                        
            for face in faces: # Para cada rosto detectado...


                # Pega as coordenadas do rosto selecionado
                (x, y, w, h) = face 
                face_x = int(x + w / 2)
                face_y = int(y + h / 2)

                # Define uma folga do centro
                folga_xPos = center_x + 50
                folga_xNeg = center_x - 50
                folga_yPos = center_y + 50
                folga_yNeg = center_y - 50

                if 'rosto' in locals(): # Verifica se existe a Varivel "Rosto" existe
                    if indice_face == rosto: # Verifica se o rosto que está sendo verificado e o que está no centro
                        # Verifica se existe algum rosto no centro
                        if face_x <= folga_xPos and face_x >= folga_xNeg and face_y <= folga_yPos and face_y >= folga_yNeg:
                        

                            cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colorb, thickness=2) # Define a elipse de posicionamento como Azul == Tirando foto

                            # Mostra quanto tempo falta para tirar a foto
                            if i % 10 == 0: 
                                i2 -= 1
                            if i2 != 0:
                                cv2.putText(frame, f'Tirando Foto em {i2}',(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorb, thickness=2)
                                
                            # Tira a foto quando o contador zerar
                            if i <= 0:
                                foto = True # Fala que a foto foi feita
                            i -= 1 # Diminui o contador
                        
                        
                        # Reseta as variaveis ao ver que o rosto não está no centro
                        else:
                            i = 50 
                            i2 = 6
                            # cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colora, thickness=2) # Desenha a elipse
                            del rosto
                else:
                    # Verifica se há um rosto no centro e envia o index do rosto no meio
                    if face_x <= folga_xPos and face_x >= folga_xNeg and face_y <= folga_yPos and face_y >= folga_yNeg:
                        cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colorb, thickness=2)
                        rosto = indice_face
                    else: # Desenha o circulo se tiver rosto detectado mas não no centro
                        cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colora, thickness=2) # Desenha a elipse
                        cv2.putText(frame, f'Posicione o Rosto no local indicado!', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colora, thickness=2)
                        # cv2.rectangle(frame, (x,y), (x + w,y + h), (0,0,0), -1)
                        cv2.rectangle(image, (x,y), (x + w,y + h), (0,0,0), -1)
                indice_face += 1
                        

                
                    
            
        else: # Desenha a elipse vermelha pois não há rostos detectados
            cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, colora, thickness=2)
            cv2.putText(frame, f'Nenhum Rosto localizado', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colora, thickness=2)
        '''
        
        cv2.imshow("Capture para tirar o bpm...", frame) # abrir a janela da Webcam com o frame atual
        
        
        k = cv2.waitKey(30) & 0xff # Detectar A tecla clicada e salvar os ultimos Bits
        
        if k == 27: # Verifica se a tecla clicada é o ESC se for para o programa
            break


        if foto: # Verifica se alguma foto foi tirada
            crop = cv2.resize(image, (1920, 1080)) # Corta a imagem e redimenciona
            cv2.imwrite('imagem.png', crop) # Escreve a imagem (salva)
            
            # print("Foto tirada!")
            foto = False # Reseta a variavel foto
            continuar = False # Para a execução do programa
            time.sleep(1.5) 

    # Destroi os processos
    captura.release()
    cv2.destroyAllWindows()
    return 'imagem.png'

    # Chama a função