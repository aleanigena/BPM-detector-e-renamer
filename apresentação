# BPM Renamer 🎚️🎼

Ferramenta desktop para análise e organização de músicas, desenvolvida especialmente para **DJs, produtores e colecionadores de música eletrônica**.

O BPM Renamer analisa arquivos de áudio, identifica o andamento da faixa, permite renomear automaticamente os arquivos com o BPM e oferece uma segunda ferramenta para análise de **espectro, tom musical e roda Camelot**.

O projeto foi pensado principalmente para gêneros eletrônicos com andamento estável e BPM elevado, como **Psytrance, Darkpsy, Forest, Hi-Tech, Psycore, Hardcore e estilos relacionados**.

---

## ✨ Principais recursos

### 🎚️ Renomeamento automático por BPM

Analisa arquivos:

* `.mp3`
* `.flac`
* `.wav`

e adiciona o BPM ao início do nome do arquivo.

Exemplo:

```text
Minha Musica.mp3
```

torna-se:

```text
148,00 - Minha Musica.mp3
```

O sistema trabalha com BPM de até **800 BPM**, permitindo inclusive trabalhar com múltiplas oitavas de andamento, como:

```text
75 BPM
150 BPM
300 BPM
```

A faixa de análise pode ser ajustada manualmente para ajudar a evitar interpretações incorretas.

---

## 🎯 Perfis de BPM

O programa possui faixas pré-configuradas para diferentes estilos:

| Perfil             |                 Faixa |
| ------------------ | --------------------: |
| Psytrance geral    |           135–200 BPM |
| Forest / Full-on   |           135–155 BPM |
| Darkpsy            |           145–175 BPM |
| Hi-Tech            |           165–195 BPM |
| Psycore / Hardcore |           190–350 BPM |
| Extremo            |           200–500 BPM |
| Personalizado      | Definido pelo usuário |

Essas faixas ajudam o algoritmo a distinguir diferentes interpretações de andamento, especialmente em músicas que podem ser detectadas em 1/2 ou 2x o BPM real.

---

## 🧠 Detecção de BPM

O algoritmo não depende simplesmente de um contador de batidas tradicional.

O sistema analisa a periodicidade do **envelope de ataques/transientes**, dando atenção especial às frequências graves associadas ao bumbo e combinando essa informação com o espectro geral da música.

A análise utiliza FFT com interpolação e harmônicos para melhorar a estabilidade da estimativa em música eletrônica de andamento fixo.

### Modos de análise

* **Preciso** — recomendado
* **Máximo** — utiliza uma janela maior
* **Rápido** — análise mais curta

O modo preciso utiliza uma janela de aproximadamente 180 segundos, enquanto os modos rápido e máximo utilizam aproximadamente 60 e 480 segundos, respectivamente.

---

## 📊 Sistema de confiança

O resultado da análise possui níveis de confiança:

* 🟢 **Alta**
* 🟠 **Média**
* 🔴 **Baixa**

O algoritmo também verifica diferentes segmentos da música para avaliar se o mesmo andamento aparece de forma consistente.

Quando existe possibilidade de confusão entre oitavas, o programa apresenta um aviso para que o usuário verifique o resultado.

---

## 🔢 Ajuste automático do BPM

Quando a detecção encontra um valor muito próximo de um andamento utilizado normalmente por produtores, o programa pode aproximá-lo para valores redondos.

Por exemplo:

```text
148,02 → 148,00
150,48 → 150,50
```

O recurso pode ser ativado ou desativado pelo usuário.

---

# 🎼 Análise de Espectro e Tom

A segunda aba do programa permite analisar uma faixa individualmente.

O sistema apresenta:

* Espectrograma;
* Frequências ao longo do tempo;
* Distribuição das notas;
* Tom estimado;
* Maior ou menor;
* Código Camelot;
* Cor correspondente à tonalidade;
* Chaves compatíveis para mixagem;
* Outras possibilidades de tonalidade.

A análise harmônica utiliza perfis de **Krumhansl-Kessler** e uma representação cromática das 12 notas.

---

## 🎨 Roda Camelot

O programa converte a tonalidade identificada para o sistema Camelot utilizado por DJs.

Exemplo:

```text
Am → 8A
C  → 8B
```

A interface apresenta visualmente a posição da faixa na roda e destaca tonalidades compatíveis para mixagem harmônica.

Também são apresentadas alternativas quando a análise possui alguma ambiguidade.

> Em músicas eletrônicas com muito grave e poucas informações melódicas, a identificação do tom pode ser naturalmente ambígua. Por isso, as alternativas devem ser tratadas como referência e não como verdade absoluta.

---

# 📈 Espectrograma

O espectrograma utiliza uma representação baseada em frequências Mel e permite visualizar a evolução do conteúdo espectral da faixa.

A interface apresenta:

* eixo de frequência;
* eixo temporal;
* visualização espectral;
* identificação de frequências relevantes;
* duração da faixa.

Isso pode ser útil para DJs e produtores que desejam observar a estrutura espectral de uma música antes de utilizá-la em um set ou processo de produção.

---

# 📁 Organização de músicas

O programa permite:

* adicionar arquivos individualmente;
* adicionar uma pasta inteira;
* incluir subpastas;
* selecionar vários arquivos;
* remover arquivos da lista;
* limpar a lista;
* editar manualmente o BPM;
* multiplicar BPM por 2;
* dividir BPM por 2.

Arquivos que já possuem BPM no início do nome são identificados automaticamente e podem ser ignorados para evitar renomeamentos desnecessários.

---

# ↩️ Desfazer

O renomeamento é realizado em lotes e o programa mantém um histórico para permitir desfazer a última operação.

Exemplo:

```text
148,00 - Minha Musica.mp3
```

pode ser restaurado para:

```text
Minha Musica.mp3
```

O sistema também evita sobrescrever um arquivo quando já existe outro com o mesmo nome.

---

# ⚙️ Tecnologia

O projeto foi desenvolvido em Python utilizando:

* **Python 3**
* **Tkinter**
* **Librosa**
* **NumPy**
* **SoundFile**
* **FFT / processamento espectral**
* **Threading**
* **Pathlib**

### Interface

A interface gráfica é construída com `Tkinter`/`ttk`, sem depender de frameworks gráficos externos.

### Processamento

As análises são executadas em threads separadas para evitar que a interface fique bloqueada durante o processamento de arquivos.

---

# 🚀 Instalação

Clone o projeto:

```bash
git clone https://github.com/SEU-USUARIO/bpm-renamer.git
cd bpm-renamer
```

Instale as dependências:

```bash
pip install librosa numpy soundfile
```

Para trabalhar com arquivos MP3, recomenda-se ter o **FFmpeg** instalado e disponível no PATH do sistema.

---

# ▶️ Executando

Execute:

```bash
python bpm_renamer_gui.py
```

A interface será aberta automaticamente.

---

# 🖥️ Fluxo de utilização

## 1. Adicione suas músicas

Escolha:

```text
Adicionar pasta...
```

ou:

```text
Adicionar arquivos...
```

Também é possível habilitar a busca em subpastas.

## 2. Escolha o estilo

Selecione uma faixa de BPM apropriada para o gênero ou configure uma faixa personalizada.

## 3. Analise

Clique:

```text
▶ Analisar BPM
```

O programa processará as músicas individualmente e mostrará:

* BPM;
* confiança;
* novo nome;
* status da análise.

## 4. Confira os resultados

Resultados com confiança baixa devem ser conferidos antes do renomeamento.

## 5. Renomeie

Clique:

```text
✔ Renomear
```

Os arquivos serão renomeados no próprio diretório.

## 6. Desfaça se necessário

Caso alguma alteração precise ser revertida:

```text
↩ Desfazer último lote
```

---

# 🎧 Para quem é este projeto?

O BPM Renamer foi pensado especialmente para:

* DJs;
* produtores musicais;
* colecionadores de música eletrônica;
* organizadores de bibliotecas musicais;
* pesquisadores de áudio;
* artistas de Psytrance;
* produtores de Darkpsy;
* DJs de Hi-Tech;
* produtores de Forest;
* artistas de Psycore/Hardcore.

O foco principal é tornar mais rápido o processo de **organizar grandes bibliotecas de música eletrônica**.

---

# ⚠️ Limitações

A detecção automática de BPM e tonalidade é uma estimativa baseada no conteúdo do áudio.

Resultados podem variar em músicas com:

* andamento variável;
* mudanças de BPM;
* polirritmia;
* intros/outros sem bateria;
* grande quantidade de efeitos;
* pouca informação harmônica;
* afinação incomum;
* forte predominância de elementos percussivos.

Em particular, músicas eletrônicas muito percussivas podem apresentar mais de uma possibilidade de tonalidade. O próprio aplicativo apresenta alternativas nesses casos.

Por isso, o programa foi projetado para **auxiliar o DJ/produtor**, e não substituir completamente a análise humana.

---

# 🛠️ Roadmap

Possíveis evoluções futuras:

* [ ] Suporte a `.ogg`, `.aiff` e outros formatos;
* [ ] Drag & Drop;
* [ ] Análise automática de BPM + Key em lote;
* [ ] Renomeamento usando BPM + Key;
* [ ] Renomeamento usando Camelot;
* [ ] Exportação de relatório CSV;
* [ ] Exportação de tags ID3;
* [ ] Escrita de BPM diretamente nos metadados;
* [ ] Escrita de Key/Camelot nos metadados;
* [ ] Detecção de energia;
* [ ] Detecção de gênero;
* [ ] Sistema de presets;
* [ ] Tema escuro;
* [ ] Empacotamento como `.exe`;
* [ ] Testes automatizados;
* [ ] Interface multilíngue.

---

# 📜 Licença

Defina aqui a licença escolhida para o projeto.

Exemplo:

```text
MIT License
```

---

# 👤 Autor

Desenvolvido para facilitar a organização e análise de bibliotecas de música eletrônica.

**BPM Renamer**
🎚️ BPM • 🎼 Key • 🎨 Camelot • 📊 Spectrum
