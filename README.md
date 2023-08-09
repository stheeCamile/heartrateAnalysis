# HeartRate Analysis

Monitor de Frequência Cardíaca usando Vídeo e Transformada de Fourier
Este repositório contém um código para medir a frequência cardíaca a partir de um vídeo usando a técnica da Transformada de Fourier. Ele captura uma foto, cria um vídeo e analisa a frequência cardíaca usando a Transformada Rápida de Fourier (FFT) para detectar o sinal do pulso.

# Como Usar
Captura da Foto: O arquivo main.py chama a função TirarFoto() para capturar uma foto. A foto capturada é salva como imagem.png.
Criação do Vídeo: A função criar_video() é chamada com a foto capturada, a duração desejada e as configurações de vídeo para criar um vídeo chamado video.mp4. O vídeo simulará a imagem capturada pelo tempo especificado.
Detecção da Frequência Cardíaca: A função startProcessing() é chamada com a taxa de quadros por segundo (fps) e o nome do vídeo (video.mp4). Ela processará os quadros do vídeo, aplicará a Transformada de Fourier e calculará a frequência cardíaca. A média de BPM, o BPM mais baixo e o BPM mais alto serão exibidos nos quadros do vídeo.

# Requisitos
Python 3.x
OpenCV
MediaPipe
NumPy
Kalman Filter

# Uso
Instale as bibliotecas necessárias usando pip install opencv-python mediapipe numpy filterpy.

Execute o arquivo main.py para capturar a foto, criar o vídeo e calcular a frequência cardíaca usando a técnica da Transformada de Fourier.
As métricas de frequência cardíaca calculadas (média de BPM, BPM mais baixo e BPM mais alto) serão exibidas nos quadros do vídeo processado.

# Observação
O código fornecido é um exemplo simplificado e pode precisar de ajustes para casos de uso ou ambientes específicos e não garante 100% de .
Certifique-se de ter as dependências necessárias instaladas e forneça caminhos/arquivos corretos ao usar as funções.
A precisão do cálculo da frequência cardíaca pode variar dependendo de vários fatores, incluindo qualidade do vídeo, condições de iluminação e estabilidade.
Este código é apenas para fins educacionais e de demonstração, não deve ser usado como dispositivo médico.
Para mais detalhes e ajustes finos, consulte o código e ajuste-o de acordo com suas necessidades.
