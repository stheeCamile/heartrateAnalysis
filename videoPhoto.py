from PIL import Image
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def criar_video(foto, duracao, nome_video, taxa_bits):
    imagem = Image.open(foto)
    largura_video, altura_video = imagem.size
    imagem_array = np.array(imagem)
    
    quadros_por_segundo = 30
    
    
    def criar_quadro(t):
        return np.fliplr(imagem_array)
    

    video = VideoClip(criar_quadro, duration=duracao)
    video.write_videofile(nome_video, fps=quadros_por_segundo, bitrate=f"{taxa_bits}k", codec="libx265",
                              ffmpeg_params=["-pix_fmt", "yuv420p"])
    
    print(f"O vídeo {nome_video} foi criado com sucesso! com taxa de bit {taxa_bits}")
    
    return

""" # Exemplo de uso
taxa_bits = 409

criar_video("imagem.png", 30, "meu_video.mp4", taxa_bits) """