# SEGMENTAÇÃO DE INSTÂNCIAS PARA ESTIMAÇÃO DO COMPRIMENTO DE PEIXES UTILIZANDO TÉCNICAS DE INTELIGÊNCIA ARTIFICIAL
Luhan Bavaresco
### Resumo
Nos últimos anos, cada vez mais cresce o interesse e a preocupação com as boas práticas e o bem-estar animal na pecuária. Na piscicultura, isso não é diferente. Apesar de não ser muito discutido, os peixes são animais sencientes e respondem rapidamente a estímulos nocivos ao bem-estar animal, podendo gerar efeitos negativos diretos e indiretos à produção. Em razão disso, um manejo adequado, com baixos condicionantes de estresse, pode ter impacto positivo na produção. Nesse contexto, este trabalho teve como objetivo desenvolver um método alternativo eficiente para estimar o comprimento de peixes. Para isso, o estudo utilizou técnicas de Inteligência Artificial (IA) como uma alternativa aos métodos tradicionais de manejo, que submetem o animal à condicionantes ambientais de estresse. Para a realização do trabalho, inicialmente, foram coletadas 679 imagens de peixes em diferentes fases de desenvolvimento. Em seguida, um modelo de segmentação de instâncias foi treinado utilizando a arquitetura YOLOv8. Com o modelo de segmentação em mãos, foi desenvolvido um código que realiza a estimativa do comprimento de cada peixe na imagem a partir das máscaras geradas pela segmentação. Esse código foi implantado na plataforma Streamlit, resultando em uma aplicação web interativa e amigável. A aplicação permite aos usuários fazer o upload de suas imagens e ter acesso a obter resultados precisos sobre a quantidade de peixes no tanque e o comprimento médio. Além disso, oferece recursos de visualização, como a exibição da segmentação e do comprimento dos peixes, juntamente com uma linha que conecta os pontos mais distantes. A análise dos resultados revelaram uma acurácia de aproximadamente 88.2% em relação aos tamanhos reais dos peixes. No estudo também verificou-se que o reflexo da luz na água e o sombreamento durante a aquisição das imagens impactam negativamente nos resultados, ressaltando a importância de se considerar esses fatores e buscar reduzi-los para obter uma estimativa de comprimento ainda mais precisa. Com base nos resultados obtidos, pode-se verificar que o modelo desenvolvido apresenta potencial para sua substituição pelo manejo tradicional, garantindo melhores práticas de bem-estar animal. Este trabalho também fornece uma base sólida para pesquisas e desenvolvimentos na área, impulsionando a piscicultura a alcançar novos patamares de excelência.
