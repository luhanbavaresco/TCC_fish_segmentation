import os
import tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

@st.cache_data()
def load_model():
    try:
        model = YOLO("./best.pt")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        model = None
    return model



def predict_fish_length(image_path, line):
    model = load_model()

    # Verificar se o arquivo de imagem existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image Not Found {image_path}")

    # Realizar previsão
    predict = model.predict(image_path)

    # Carregar a imagem original
    imagem_original = cv2.imread(image_path)

    # Inicializar variáveis
    media_cm_tanque = 0
    contador_peixes = 0
    comprimento_list = []
    # Processar cada máscara segmentada
    for i, mascara in enumerate(predict[0].masks.xyn):
        # Obter as coordenadas x e y da máscara
        x = (mascara[:, 0] * imagem_original.shape[1]).astype("int")
        y = (mascara[:, 1] * imagem_original.shape[0]).astype("int")

        # Verificar se as dimensões das máscaras correspondem à imagem original
        if x.size > 0 and y.size > 0:
            # Encontrar os dois pontos mais distantes em paralelo
            distancia_maxima = 0
            ponto1 = None
            ponto2 = None

            for j in range(len(x)):
                for k in range(j + 1, len(x)):
                    distancia = np.linalg.norm(
                        np.array([x[j], y[j]]) - np.array([x[k], y[k]])
                    )
                    if distancia > distancia_maxima:
                        distancia_maxima = distancia
                        ponto1 = np.array([x[j], y[j]])
                        ponto2 = np.array([x[k], y[k]])

            # Calcular o comprimento em centímetros
            relacao_pixels_centimetros = 257  # Relação entre pixels e centímetros
            # Tamanho em centímetros correspondente a 'relacao_pixels_centimetros'
            tamanho_centimetros = 15.2

            # Distância em pixels calculada anteriormente
            distancia_pixels = distancia_maxima
            distancia_centimetros = (distancia_pixels / relacao_pixels_centimetros) * tamanho_centimetros

            # Calcular o ponto médio entre os dois pontos
            ponto_medio = (ponto1 + ponto2) // 2
            if line == 1:
                # Desenhar uma linha entre os dois pontos mais distantes
                cv2.line(imagem_original, tuple(ponto1), tuple(ponto2), (127, 255, 212), 2)
                contador_peixes += 1
                media_cm_tanque += distancia_centimetros
                comprimento_list.append(distancia_centimetros)
            else:
                # Exibir o comprimento em centímetros na imagem original
                if distancia_centimetros > 5.0:
                    texto = f"{distancia_centimetros:.2f} cm"
                    tamanho_texto, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 2)
                    ponto_texto = (ponto_medio[0] - tamanho_texto[0] // 2,ponto_medio[1] - 10,)
                    cv2.putText(imagem_original,texto, ponto_texto,cv2.FONT_HERSHEY_TRIPLEX,0.6,(0, 0, 200),2,)
                    contador_peixes += 1
                    media_cm_tanque += distancia_centimetros
                    comprimento_list.append(distancia_centimetros)

            # Definir um valor de canal alfa de 0 para a cor de preenchimento (verde claro totalmente transparente)
            # Cor para o preenchimento da segmentação do objeto (verde claro totalmente transparente)
            cor_preenchimento = (0, 0, 50, 0)

            # Criar uma imagem em branco com o mesmo tamanho e tipo da imagem original
            imagem_transparente = np.zeros_like(imagem_original, dtype=np.uint8)

            # Desenhar a segmentação do YOLOv8 na imagem transparente
            cv2.polylines(imagem_transparente,[np.vstack((x, y)).T],isClosed=True,color=cor_preenchimento,thickness=2,)
            cv2.fillPoly(imagem_transparente, [np.vstack((x, y)).T], color=cor_preenchimento)

            # Compor a imagem transparente sobre a imagem original
            imagem_original = cv2.add(imagem_original, imagem_transparente)

    # Calcular a média do comprimento dos peixes
    peixes_tanque = f"{media_cm_tanque / contador_peixes:.2f}"
    #peixes_tanque = 0
    # Salvar a imagem com as distâncias em um arquivo temporário
    output_file = os.path.join(os.getcwd(), "output.jpg")
    # cv2.imwrite(output_file, imagem_original)
    imagem_salva = Image.fromarray(imagem_original)
    imagem_salva.save(output_file)

    return comprimento_list, output_file, contador_peixes, peixes_tanque
def main():
    st.header("SEGMENTAÇÃO DE INSTÂNCIAS PARA ESTIMAÇÃO DO COMPRIMENTO DE PEIXES\n\n:fish::blowfish::tropical_fish::straight_ruler:")
    st.write("Por Luhan Bavaresco")

    st.caption("**OBSERVAÇÃO:**\n\n*Para uma maior precisão na estimativa inteligente de comprimento de peixes, deve-se seguir as observações abaixo:*\n\n\t1. Altura da água no tanque: 16.5 cm\n\n\t2. Distância entre câmera e o fundo do tanque: 78 cm\n\n\t3. Imagem sem reflexo da luz na água\n\n\t4. Imagem sem sombreamento")

    uploaded_file = st.file_uploader("\n\nEscolha uma imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Salvar a imagem em um arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            image_path = temp_file.name

        # Realizar a previsão e obter o resultado
        comprimento_list, output_file, contador_peixes, peixes_tanque = predict_fish_length(image_path, 0)

        # Exibir informações e a imagem de saída
        st.write("Quantidade de peixes no tanque:", int(contador_peixes))
        st.write("Comprimento médio no tanque:", float(peixes_tanque), "cm")

        # Verificar se o arquivo temporário existe
        if os.path.exists(image_path):
            image = open(image_path, "rb").read()
            desc = "Imagem original"
            last_image = None  # Variável para armazenar a última imagem exibida
            last_desc = None
            # Botões de seleção de visualização
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Segmentação e comprimento"):
                    # Código para mostrar apenas uma linha
                    comprimento_list, output_file, contador_peixes, peixes_tanque = predict_fish_length(image_path, 0)
                    image = open(output_file, "rb").read()
                    desc = "Peixes segmentados e seu comprimento em cm"
                    # st.image(image, caption="Peixes segmentados e uma linha entre os dois pontos", use_column_width=True, clamp=True)

            with col2:
                if st.button("Segmentação e linha mágica"):
                    # Código para mostrar apenas uma linha
                    comprimento_list, output_file, contador_peixes, peixes_tanque = predict_fish_length(image_path, 1)
                    image = open(output_file, "rb").read()
                    desc = "Peixes segmentados e a linha do comprimento"
                    # st.image(image, caption="Peixes segmentados e uma linha entre os dois pontos", use_column_width=True, clamp=True)

            with col3:
                if st.button("Imagem original"):
                    image = open(image_path, "rb").read()
                    desc = "Imagem original"


            # Imagem inicial de segmentação e comprimento
            st.image(image, caption=desc, use_column_width=True, clamp=True)
            
            # Gerar o histograma
            hist_values, hist_bins = np.histogram(comprimento_list, bins=6)

            # Criar uma sequência de cores degradê de ciano
            colors = plt.cm.cool(np.linspace(0.2, 0.8, len(hist_values)))

            # Plot do histograma com cores degradê de ciano
            plt.bar(hist_bins[:-1], hist_values, width=np.diff(hist_bins), color=colors)

            # Configurar os rótulos e título
            plt.xlabel('Comprimento (cm)')
            plt.ylabel('Frequência')
            plt.title('Distribuição do Comprimento dos Peixes')

            # Exibir o histograma
            st.pyplot(plt)


        else:
            st.write("Arquivo de saída não encontrado.")

        # Remover o arquivo temporário
        os.remove(image_path)
        os.remove(output_file)


if __name__ == "__main__":
    main()
