from tirarFoto import TirarFoto
from videoPhoto import criar_video
from mainOrigin import startProcessing
from os import remove, path

TirarFoto()
if path.isfile('./imagem.png'):
    nome_foto = 'imagem.png'
    taxa_bits = 409
    nome_video = 'video.mp4'
    duracao = 30
    criar_video(nome_foto, duracao, nome_video, taxa_bits)
    startProcessing(30, nome_video)
    remove('imagem.png')