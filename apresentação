#!/usr/bin/env python3
"""
BPM Renamer v4

Aba 1 - Renomear por BPM: detecta o BPM (2 casas decimais) de músicas mp3, flac e wav,
        otimizado para música eletrônica de andamento estável e BPM alto
        (psytrance, darkpsy, hi-tech, forest, psycore... de 135 a 500 BPM),
        e renomeia os arquivos colocando o BPM na frente do nome.

            "Minha Musica.mp3"  ->  "148,00 - Minha Musica.mp3"

Aba 2 - Espectro e Tom: mostra o espectrograma da faixa, o tom (nota + maior/menor)
        e o código na roda Camelot, com a cor correspondente e as chaves compatíveis.

Instalação:
    pip install librosa numpy soundfile
    (para mp3 é recomendável ter o ffmpeg instalado)

Uso:
    python bpm_renamer_gui.py
"""

import colorsys
import math
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

EXTENSOES = {".mp3", ".flac", ".wav"}
SEP_DECIMAL = ","  # troque por "." se preferir 148.00

# Reconhece nomes que já começam com BPM: "148 - ", "148,5 - ", "148,53 - ", "148.53 - "
PADRAO_JA_RENOMEADO = re.compile(r"^\d{2,3}(?:[.,]\d{1,2})?\s*-\s")

# Estilos: faixa de BPM esperada (ajuda a escolher a oitava certa: 150 x 300 x 75)
ESTILOS = {
    "Psytrance geral (135–200)": (135, 200),
    "Forest / Full-on (135–155)": (135, 155),
    "Darkpsy (145–175)": (145, 175),
    "Hi-Tech (165–195)": (165, 195),
    "Psycore / Hardcore (190–350)": (190, 350),
    "Extremo (200–500)": (200, 500),
    "Personalizado": None,
}

# Modo -> duração (s) do trecho analisado (do meio da música)
MODOS = {
    "Preciso (recomendado)": "preciso",
    "Máximo (música inteira)": "maximo",
    "Rápido": "rapido",
}
JANELAS = {"rapido": 60.0, "preciso": 180.0, "maximo": 480.0}

HOP = 128        # ~5,8 ms por quadro a 22050 Hz
N_FFT = 1024
PESOS_HARMONICOS = (1.0, 0.6, 0.4, 0.3)


def fmt_bpm(bpm):
    return f"{bpm:.2f}".replace(".", SEP_DECIMAL)


# --------------------------------------------------------------------------- #
#  Detecção de BPM
#
#  Em vez de rastrear batida por batida, o programa mede a periodicidade do
#  "envelope de ataques" (onde estão os bumbos e transientes) com uma FFT bem
#  interpolada. Em música eletrônica, feita em DAW com andamento fixo, isso
#  chega a centésimos de BPM e funciona bem em andamentos muito altos.
# --------------------------------------------------------------------------- #
def calcular_envelope(y, sr):
    """Envelope de ataques: fluxo espectral da faixa grave (bumbo) + faixa total."""
    import librosa
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    sel = freqs <= 8000
    f_sel = freqs[sel]
    grave = f_sel <= 250
    total = f_sel >= 30

    bloco = HOP * 8000  # processa em blocos (~46 s) para economizar memória
    fl_grave, fl_total = [], []
    prev = None
    pos = 0
    while pos + N_FFT <= len(y):
        seg = y[pos: pos + bloco + N_FFT - HOP]
        if len(seg) < N_FFT:
            break
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, center=False))[sel]
        L = 20.0 * np.log10(np.maximum(S, 1e-4))          # dB, referência fixa
        ext = np.concatenate([L[:, :1] if prev is None else prev, L], axis=1)
        d = np.maximum(np.diff(ext, axis=1), 0.0)          # só aumentos de energia
        fl_grave.append(d[grave].mean(axis=0))
        fl_total.append(d[total].mean(axis=0))
        prev = L[:, -1:]
        pos += L.shape[1] * HOP

    g = np.concatenate(fl_grave)
    t = np.concatenate(fl_total)
    return g / (g.std() + 1e-9) + 0.5 * t / (t.std() + 1e-9)


def _passa_alta(x, janela):
    import numpy as np
    janela = max(3, int(janela) | 1)
    return x - np.convolve(x, np.ones(janela) / janela, mode="same")


def estimar_tempo_espectral(env, fs, minimo, maximo):
    """
    Acha o andamento dentro de [minimo, maximo] BPM pelo pico do espectro do
    envelope, somando os harmônicos (2x, 3x, 4x) para firmar o resultado.

    Retorna (bpm, proeminência do pico, ambiguidade de oitava).
    """
    import numpy as np

    x = _passa_alta(np.asarray(env, dtype=float), 1.5 * fs)
    x = x * np.hanning(len(x))
    N = 1 << int(np.ceil(np.log2(len(x) * 32)))   # zero-padding: ~0,01 BPM por ponto
    X = np.abs(np.fft.rfft(x, N))
    df = fs / N * 60.0                             # BPM por ponto do espectro

    def H_bin(i):
        return sum(w * X[i * h] for h, w in enumerate(PESOS_HARMONICOS, 1) if i * h < len(X))

    i0 = max(1, int(np.floor(minimo / df)))
    i1 = min(int(np.ceil(maximo / df)), len(X) - 1)
    idx = np.arange(i0, i1 + 1)
    H = np.zeros(len(idx))
    for h, w in enumerate(PESOS_HARMONICOS, 1):
        j = idx * h
        ok = j < len(X)
        H[ok] += w * X[j[ok]]

    i = int(idx[int(np.argmax(H))])
    # Faixa larga: prefere a oitava mais baixa (ritmo do bumbo) se tiver energia comparável
    while i // 2 >= i0 and H_bin(i // 2) >= 0.5 * H_bin(i):
        i //= 2

    # Refinamento parabólico em torno do pico
    a, b, c = H_bin(i - 1), H_bin(i), H_bin(i + 1)
    den = a - 2 * b + c
    delta = float(np.clip(0.5 * (a - c) / den, -1, 1)) if den != 0 else 0.0
    bpm = (i + delta) * df

    # Há outra oitava (metade/dobro) dentro da faixa com energia relevante?
    alt = []
    if i // 2 >= i0:
        alt.append(H_bin(i // 2) / b)
    if i * 2 <= i1:
        alt.append(H_bin(i * 2) / b)
    ambiguidade = max(alt) if alt else 0.0
    return bpm, float(b / (np.median(H) + 1e-12)), ambiguidade


def analisar_envelope(env, fs, minimo, maximo, segmentos=True):
    """Retorna (bpm, confiança, aviso). Confiança: 'Alta' | 'Média' | 'Baixa'."""
    bpm, prom, amb = estimar_tempo_espectral(env, fs, minimo, maximo)

    # Verificação cruzada: o mesmo andamento aparece em 3 trechos da música?
    spread = None
    n = len(env) // 3
    if segmentos and n >= int(20 * fs):
        ests = [
            estimar_tempo_espectral(env[k * n:(k + 1) * n], fs, bpm * 0.985, bpm * 1.015)[0]
            for k in range(3)
        ]
        spread = max(ests) - min(ests)

    aviso = "verificar oitava" if amb > 0.3 else ""
    if amb > 0.3 or prom < 3:
        conf = "Baixa"
    elif spread is None:
        conf = "Média" if prom >= 4 else "Baixa"
    elif spread <= 0.2 and prom >= 6:
        conf = "Alta"
    elif spread <= 0.6:
        conf = "Média"
    else:
        conf = "Baixa"
    return bpm, conf, aviso


def encaixar_valor(bpm, tolerancia=0.06):
    """Produtores costumam usar andamentos redondos (145,00 / 145,50): encaixa se estiver muito perto."""
    cand = round(bpm * 2) / 2
    return cand if abs(bpm - cand) <= tolerancia else bpm


def detectar_bpm(caminho, minimo, maximo, modo="preciso", encaixar=True):
    """Retorna (bpm com 2 casas, confiança, aviso)."""
    import librosa

    janela = JANELAS.get(modo, 180.0)
    try:
        dur = float(librosa.get_duration(path=str(caminho)))
    except TypeError:  # versões antigas do librosa
        dur = float(librosa.get_duration(filename=str(caminho)))

    if dur > janela:  # trecho do meio: evita introdução/final sem batida marcada
        y, sr = librosa.load(str(caminho), sr=22050, mono=True,
                             offset=(dur - janela) / 2, duration=janela)
    else:
        y, sr = librosa.load(str(caminho), sr=22050, mono=True)

    if len(y) < sr * 15:
        raise ValueError("arquivo muito curto para medir o BPM (mínimo ~15 s)")

    env = calcular_envelope(y, sr)
    bpm, conf, aviso = analisar_envelope(env, sr / HOP, minimo, maximo, segmentos=(modo != "rapido"))
    if encaixar:
        bpm = encaixar_valor(bpm)
    return round(bpm, 2), conf, aviso


# --------------------------------------------------------------------------- #
#  Tom musical, roda Camelot e cores
# --------------------------------------------------------------------------- #
# Perfis de Krumhansl-Kessler (peso de cada nota em relação à tônica)
KK_MAIOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MENOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

NOMES_MAIOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NOMES_MENOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
NOMES_PT = {"C": "Dó", "D": "Ré", "E": "Mi", "F": "Fá", "G": "Sol", "A": "Lá", "B": "Si"}
NOTAS_BARRAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Número Camelot de cada tônica (índice 0 = Dó). Maior = letra B, menor = letra A.
CAMELOT_MAIOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
CAMELOT_MENOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]

# Matiz (graus) de cada número da roda, do 1 (turquesa) ao 12 (azul claro)
CAMELOT_MATIZ = {1: 170, 2: 135, 3: 100, 4: 60, 5: 35, 6: 10,
                 7: 350, 8: 325, 9: 295, 10: 265, 11: 235, 12: 205}


def codigo_camelot(pc, menor):
    return f"{(CAMELOT_MENOR if menor else CAMELOT_MAIOR)[pc]}{'A' if menor else 'B'}"


def nome_tom(pc, menor):
    """Retorna ('Lá menor', 'Am')."""
    letra = (NOMES_MENOR if menor else NOMES_MAIOR)[pc]
    pt = NOMES_PT[letra[0]] + letra[1:]
    return f"{pt} {'menor' if menor else 'maior'}", f"{letra}{'m' if menor else ''}"


def cor_camelot(codigo):
    """Cor (hex) da chave: mesmo matiz para A e B do mesmo número; B mais clara, A mais intensa."""
    num, letra = int(codigo[:-1]), codigo[-1]
    s, v = (0.55, 0.97) if letra == "B" else (0.80, 0.86)
    r, g, b = colorsys.hsv_to_rgb(CAMELOT_MATIZ[num] / 360.0, s, v)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def texto_sobre(cor_hex):
    """Preto ou branco, o que ficar mais legível sobre a cor."""
    r, g, b = (int(cor_hex[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def misturar(cor_hex, fracao_branco):
    r, g, b = (int(cor_hex[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * fracao_branco) for c in (r, g, b))
    return "#%02x%02x%02x" % (r, g, b)


def compativeis(codigo):
    """Chaves que combinam na mixagem harmônica: mesma, relativa (A/B) e vizinhas (±1)."""
    num, letra = int(codigo[:-1]), codigo[-1]
    outra = "B" if letra == "A" else "A"
    return {f"{num}{outra}", f"{num % 12 + 1}{letra}", f"{(num - 2) % 12 + 1}{letra}"}


def rankear_tons(chroma):
    """Correlaciona o perfil de notas (chroma) com os 24 tons. Retorna [(correlação, pc, menor)] ordenado."""
    import numpy as np

    c = np.asarray(chroma, dtype=float)
    res = []
    for pc in range(12):
        for menor, perfil in ((False, KK_MAIOR), (True, KK_MENOR)):
            r = np.corrcoef(c, np.roll(np.asarray(perfil, dtype=float), pc))[0, 1]
            res.append((float(r), pc, menor))
    res.sort(reverse=True)
    return res


def calcular_chroma(y, sr):
    """Distribuição de energia nas 12 notas (Dó=0), usando só a parte harmônica e corrigindo a afinação."""
    import librosa
    import numpy as np

    tuning = librosa.estimate_tuning(y=y, sr=sr)
    yh = librosa.effects.harmonic(y, margin=3.0)   # reduz a influência de bumbo/percussão
    hop = 2048
    C = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C2"),
                                   n_octaves=6, tuning=tuning)
    rms = librosa.feature.rms(y=yh, frame_length=4096, hop_length=hop)[0]
    n = min(C.shape[1], len(rms))
    pesos = rms[:n] ** 2                            # trechos mais fortes contam mais
    return (C[:, :n] * pesos).sum(axis=1) / (pesos.sum() + 1e-12)


# --------------------------------------------------------------------------- #
#  Espectrograma
# --------------------------------------------------------------------------- #
_LUT = None


def criar_lut():
    """Paleta de cores (preto -> roxo -> laranja -> amarelo claro) com 256 níveis."""
    global _LUT
    if _LUT is None:
        import numpy as np
        pontos = np.linspace(0, 1, 8)
        cores = np.array([(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
                          (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)], dtype=float)
        x = np.linspace(0, 1, 256)
        _LUT = np.stack([np.interp(x, pontos, cores[:, i]) for i in range(3)], axis=1).astype(np.uint8)
    return _LUT


def calcular_espectrograma(y, sr, n_mels=128, hop=1024, max_colunas=2400):
    """Retorna (matriz uint8 [n_mels x colunas], frequências dos canais mel)."""
    import librosa
    import numpy as np

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=n_mels,
                                       fmax=sr / 2, power=2.0)
    Sdb = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    n = Sdb.shape[1]
    if n > max_colunas:                      # reduz colunas (média) para manter leve
        f = int(np.ceil(n / max_colunas))
        n2 = (n // f) * f
        Sdb = Sdb[:, :n2].reshape(n_mels, n2 // f, f).mean(axis=2)
    spec = (np.clip((Sdb + 80.0) / 80.0, 0.0, 1.0) * 255).astype(np.uint8)
    mel = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0.0, fmax=sr / 2)[1:-1]
    return spec, mel


def renderizar_ppm(spec, largura, altura):
    """Redimensiona o espectrograma para (largura x altura) e devolve bytes PPM para o Tk."""
    import numpy as np

    n_mels, n_cols = spec.shape
    linhas = np.linspace(n_mels - 1, 0, altura).round().astype(int)   # graves embaixo
    colunas = np.linspace(0, n_cols - 1, largura).round().astype(int)
    rgb = criar_lut()[spec[np.ix_(linhas, colunas)]]
    return b"P6 %d %d 255\n" % (largura, altura) + rgb.tobytes()


def analisar_faixa(caminho, progresso=None):
    """Carrega a faixa e devolve espectrograma + tom. 'progresso' recebe mensagens de texto."""
    import librosa

    def avisar(msg):
        if progresso:
            progresso(msg)

    avisar("Carregando o áudio...")
    y, sr = librosa.load(str(caminho), sr=22050, mono=True, duration=720)
    if len(y) < sr * 10:
        raise ValueError("arquivo muito curto (mínimo ~10 s)")

    avisar("Gerando o espectro...")
    spec, mel = calcular_espectrograma(y, sr)

    avisar("Analisando o tom (pode levar alguns segundos)...")
    janela = 240 * sr
    ytom = y[(len(y) - janela) // 2:(len(y) - janela) // 2 + janela] if len(y) > janela else y
    chroma = calcular_chroma(ytom, sr)
    return {"spec": spec, "mel": mel, "dur": len(y) / sr, "chroma": chroma,
            "ranking": rankear_tons(chroma)[:4]}


# --------------------------------------------------------------------------- #
#  Interface
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BPM Renamer")
        self.geometry("1180x780")
        self.minsize(980, 620)

        self.items = {}          # iid -> {"path": Path, "bpm": float|None, "state": str}
        self.paths_set = set()
        self.undo_stack = []     # cada lote = [(iid, novo_path, antigo_path)]
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.running = False
        self._aplicando_estilo = False

        # aba de espectro
        self.esp_caminho = None
        self.esp_rodando = False
        self.spec = None
        self.spec_mel = None
        self.spec_dur = 0.0
        self.img_spec = None
        self._redesenho_id = None

        self._estilo()
        self._montar()
        self.protocol("WM_DELETE_WINDOW", self._fechar)
        self.after(100, self._processar_fila)

    # ---------------------------- construção da UI ------------------------- #
    def _estilo(self):
        st = ttk.Style(self)
        for tema in ("vista", "clam"):
            if tema in st.theme_names():
                st.theme_use(tema)
                break
        st.configure("Titulo.TLabel", font=("Segoe UI", 16, "bold"))
        st.configure("Sub.TLabel", foreground="#555555")
        st.configure("Acao.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        st.configure("Treeview", rowheight=26)
        fundo = st.lookup("TFrame", "background")
        self.cor_fundo = fundo or "#f0f0f0"

    def _montar(self):
        raiz = ttk.Frame(self, padding=12)
        raiz.pack(fill="both", expand=True)

        ttk.Label(raiz, text="BPM Renamer", style="Titulo.TLabel").pack(anchor="w", pady=(0, 6))

        self.abas = ttk.Notebook(raiz)
        self.abas.pack(fill="both", expand=True)
        tab1 = ttk.Frame(self.abas, padding=(2, 10, 2, 2))
        tab2 = ttk.Frame(self.abas, padding=(2, 10, 2, 2))
        self.abas.add(tab1, text="  🎚  Renomear por BPM  ")
        self.abas.add(tab2, text="  🎼  Espectro e Tom  ")

        self._montar_aba_bpm(tab1)
        self._montar_aba_espectro(tab2)

        # Progresso / status (compartilhado)
        self.progresso = ttk.Progressbar(raiz, mode="determinate")
        self.progresso.pack(fill="x", pady=(10, 2))
        self.var_status = tk.StringVar(value="Pronto. Adicione músicas para começar.")
        ttk.Label(raiz, textvariable=self.var_status, style="Sub.TLabel").pack(anchor="w")

    # -------------------------- aba 1: renomear por BPM -------------------- #
    def _montar_aba_bpm(self, raiz):
        ttk.Label(
            raiz,
            text="1) Adicione músicas   →   2) Analisar BPM   →   3) Confira e Renomear",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        # Passo 1
        f1 = ttk.LabelFrame(raiz, text=" 1. Adicionar músicas e ajustes ", padding=8)
        f1.pack(fill="x")

        l1 = ttk.Frame(f1)
        l1.pack(fill="x")
        self.btn_pasta = ttk.Button(l1, text="📁  Adicionar pasta...", command=self.adicionar_pasta)
        self.btn_pasta.pack(side="left")
        self.btn_arqs = ttk.Button(l1, text="🎵  Adicionar arquivos...", command=self.adicionar_arquivos)
        self.btn_arqs.pack(side="left", padx=6)
        self.var_sub = tk.BooleanVar(value=True)
        ttk.Checkbutton(l1, text="Incluir subpastas", variable=self.var_sub).pack(side="left", padx=10)

        l2 = ttk.Frame(f1)
        l2.pack(fill="x", pady=(8, 0))
        ttk.Label(l2, text="Estilo").pack(side="left")
        self.var_estilo = tk.StringVar(value="Psytrance geral (135–200)")
        cb = ttk.Combobox(l2, textvariable=self.var_estilo, values=list(ESTILOS), state="readonly", width=28)
        cb.pack(side="left", padx=(4, 10))
        cb.bind("<<ComboboxSelected>>", self._aplicar_estilo)

        ttk.Label(l2, text="BPM entre").pack(side="left")
        self.var_min = tk.IntVar(value=135)
        self.var_max = tk.IntVar(value=200)
        ttk.Spinbox(l2, from_=30, to=800, width=5, textvariable=self.var_min).pack(side="left", padx=4)
        ttk.Label(l2, text="e").pack(side="left")
        ttk.Spinbox(l2, from_=30, to=800, width=5, textvariable=self.var_max).pack(side="left", padx=4)
        self.var_min.trace_add("write", self._faixa_editada)
        self.var_max.trace_add("write", self._faixa_editada)

        ttk.Separator(l2, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(l2, text="Modo").pack(side="left")
        self.var_modo = tk.StringVar(value="Preciso (recomendado)")
        ttk.Combobox(l2, textvariable=self.var_modo, values=list(MODOS), state="readonly",
                     width=22).pack(side="left", padx=4)

        l3 = ttk.Frame(f1)
        l3.pack(fill="x", pady=(8, 0))
        self.var_snap = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            l3, text="Encaixar em valores redondos próximos (ex.: 148,02 → 148,00; 150,48 → 150,50)",
            variable=self.var_snap,
        ).pack(side="left")
        ttk.Label(
            f1,
            text="Dica: escolha o estilo da sua música. A faixa de BPM ajuda o programa a distinguir 150 de 300 ou 75.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        # Tabela
        meio = ttk.Frame(raiz)
        meio.pack(fill="both", expand=True, pady=10)
        cols = ("arquivo", "pasta", "bpm", "conf", "novo", "status")
        self.tree = ttk.Treeview(meio, columns=cols, show="headings", selectmode="extended")
        cab = {"arquivo": ("Arquivo", 240), "pasta": ("Pasta", 160), "bpm": ("BPM", 80),
               "conf": ("Confiança", 80), "novo": ("Novo nome", 270), "status": ("Status", 130)}
        for c, (texto, larg) in cab.items():
            self.tree.heading(c, text=texto)
            self.tree.column(c, width=larg, anchor="center" if c in ("bpm", "conf", "status") else "w")
        sb = ttk.Scrollbar(meio, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.tree.tag_configure("ready", foreground="#0a6b2d")
        self.tree.tag_configure("low", foreground="#a86400")
        self.tree.tag_configure("done", foreground="#777777")
        self.tree.tag_configure("error", foreground="#b00020")
        self.tree.tag_configure("skip", foreground="#999999")
        self.tree.bind("<Double-1>", self.editar_bpm)
        self.tree.bind("<Delete>", lambda e: self.remover_selecionados())
        self.tree.bind("<Control-a>", lambda e: (self.selecionar_todos(), "break")[1])

        # Ferramentas da lista
        f2 = ttk.Frame(raiz)
        f2.pack(fill="x")
        ttk.Button(f2, text="Selecionar todos", command=self.selecionar_todos).pack(side="left")
        ttk.Button(f2, text="Remover selecionados", command=self.remover_selecionados).pack(side="left", padx=6)
        ttk.Button(f2, text="Limpar lista", command=self.limpar_lista).pack(side="left")
        ttk.Separator(f2, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(f2, text="BPM × 2", width=9, command=lambda: self.multiplicar(2.0)).pack(side="left")
        ttk.Button(f2, text="BPM ÷ 2", width=9, command=lambda: self.multiplicar(0.5)).pack(side="left", padx=6)
        ttk.Label(f2, text="Dois cliques na linha = digitar o BPM.", style="Sub.TLabel").pack(side="right")

        # Passos 2 e 3
        f3 = ttk.Frame(raiz)
        f3.pack(fill="x", pady=(10, 0))
        self.btn_analisar = ttk.Button(f3, text="2.  ▶  Analisar BPM", style="Acao.TButton",
                                       command=self.analisar)
        self.btn_analisar.pack(side="left")
        self.btn_cancelar = ttk.Button(f3, text="⏹  Cancelar", command=self.cancelar, state="disabled")
        self.btn_cancelar.pack(side="left", padx=6)
        self.btn_renomear = ttk.Button(f3, text="3.  ✔  Renomear", style="Acao.TButton",
                                       command=self.renomear)
        self.btn_renomear.pack(side="left", padx=(24, 0))
        self.btn_desfazer = ttk.Button(f3, text="↩  Desfazer último lote", command=self.desfazer,
                                       state="disabled")
        self.btn_desfazer.pack(side="left", padx=6)

    # ------------------------ aba 2: espectro e tom ------------------------ #
    def _montar_aba_espectro(self, pai):
        topo = ttk.Frame(pai)
        topo.pack(fill="x")
        ttk.Button(topo, text="📂  Escolher arquivo...", command=self.esp_escolher).pack(side="left")
        ttk.Button(topo, text="Usar o selecionado na aba Renomear",
                   command=self.esp_usar_selecionado).pack(side="left", padx=6)
        self.btn_esp = ttk.Button(topo, text="▶  Analisar espectro e tom", style="Acao.TButton",
                                  command=self.esp_analisar)
        self.btn_esp.pack(side="left", padx=(14, 0))

        self.var_esp_arq = tk.StringVar(value="Nenhum arquivo escolhido.")
        ttk.Label(pai, textvariable=self.var_esp_arq, style="Sub.TLabel").pack(anchor="w", pady=(8, 6))

        self.pb_esp = ttk.Progressbar(pai, mode="indeterminate")
        self.pb_esp.pack(side="bottom", fill="x", pady=(8, 0))

        corpo = ttk.Frame(pai)
        corpo.pack(fill="both", expand=True)

        # Esquerda: espectrograma
        self.cv_spec = tk.Canvas(corpo, bg="#0b0b10", highlightthickness=0)
        self.cv_spec.pack(side="left", fill="both", expand=True)
        self.cv_spec.bind("<Configure>", self._agendar_redesenho)

        # Direita: painel do tom
        dir_ = ttk.Frame(corpo, padding=(14, 0, 0, 0))
        dir_.pack(side="right", fill="y")

        ttk.Label(dir_, text="Tom da faixa", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.lbl_camelot = tk.Label(dir_, text="—", font=("Segoe UI", 40, "bold"), width=6,
                                    bg="#c8c8c8", fg="#555555")
        self.lbl_camelot.pack(fill="x", pady=(4, 4))
        self.var_tom = tk.StringVar(value="Analise uma faixa para ver o tom.")
        ttk.Label(dir_, textvariable=self.var_tom, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.var_conf_tom = tk.StringVar(value="")
        ttk.Label(dir_, textvariable=self.var_conf_tom, style="Sub.TLabel").pack(anchor="w")

        ttk.Label(dir_, text="Roda Camelot", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.cv_roda = tk.Canvas(dir_, width=260, height=260, bg=self.cor_fundo, highlightthickness=0)
        self.cv_roda.pack()
        self.var_comp = tk.StringVar(value="")
        ttk.Label(dir_, textvariable=self.var_comp, style="Sub.TLabel", wraplength=270).pack(anchor="w")

        ttk.Label(dir_, text="Notas mais presentes", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self.cv_chroma = tk.Canvas(dir_, width=270, height=90, bg=self.cor_fundo, highlightthickness=0)
        self.cv_chroma.pack()

        ttk.Label(dir_, text="Outras possibilidades", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self.chips = []
        for _ in range(3):
            lb = tk.Label(dir_, text="", anchor="w", padx=8, pady=2, font=("Segoe UI", 9, "bold"))
            lb.pack(fill="x", pady=1)
            self.chips.append(lb)

        ttk.Label(
            dir_,
            text="Em música eletrônica com muito grave e poucas melodias\n(psytrance, darkpsy), o tom pode ser ambíguo.\n"
                 "Use as alternativas como referência.",
            style="Sub.TLabel", justify="left",
        ).pack(anchor="w", pady=(8, 0))

        self._desenhar_roda(None)
        self._desenhar_chroma(None)
        self._desenhar_espectro()

    # ------------------------------ ações da aba 2 ------------------------- #
    def esp_escolher(self):
        arq = filedialog.askopenfilename(
            title="Escolha uma música",
            filetypes=[("Áudio", "*.mp3 *.flac *.wav"), ("Todos os arquivos", "*.*")],
        )
        if arq:
            self.esp_caminho = Path(arq)
            self.var_esp_arq.set(f"Arquivo: {self.esp_caminho.name}")

    def esp_usar_selecionado(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Espectro e Tom", "Selecione uma música na lista da aba \"Renomear por BPM\" primeiro.")
            return
        self.esp_caminho = self.items[sel[0]]["path"]
        self.var_esp_arq.set(f"Arquivo: {self.esp_caminho.name}")

    def esp_analisar(self):
        if self.esp_rodando:
            return
        if self.esp_caminho is None:
            sel = self.tree.selection()
            if sel:
                self.esp_caminho = self.items[sel[0]]["path"]
                self.var_esp_arq.set(f"Arquivo: {self.esp_caminho.name}")
            else:
                messagebox.showinfo("Espectro e Tom", "Escolha um arquivo primeiro.")
                return
        if not self.esp_caminho.exists():
            messagebox.showwarning("Espectro e Tom", "Esse arquivo não foi encontrado (foi movido ou renomeado?).")
            return
        self.esp_rodando = True
        self.btn_esp.configure(state="disabled")
        self.pb_esp.start(12)
        self.var_status.set("Analisando espectro e tom...")
        threading.Thread(target=self._worker_espectro, args=(self.esp_caminho,), daemon=True).start()

    def _worker_espectro(self, caminho):
        try:
            import librosa  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.q.put(("esp_err", "Faltam bibliotecas.\nInstale com:\n\npip install librosa numpy soundfile"))
            return
        try:
            res = analisar_faixa(caminho, progresso=lambda m: self.q.put(("esp_status", m)))
            self.q.put(("esp_ok", res, caminho.name))
        except Exception as e:  # arquivo corrompido, sem ffmpeg, muito curto etc.
            self.q.put(("esp_err", str(e)))

    def _esp_finalizar(self):
        self.esp_rodando = False
        self.btn_esp.configure(state="normal")
        self.pb_esp.stop()

    def _mostrar_resultado_esp(self, res, nome):
        self.spec = res["spec"]
        self.spec_mel = res["mel"]
        self.spec_dur = res["dur"]
        self._desenhar_espectro(titulo=nome)

        corr0, pc, menor = res["ranking"][0]
        corr1 = res["ranking"][1][0]
        cod = codigo_camelot(pc, menor)
        cor = cor_camelot(cod)
        nome_pt, sigla = nome_tom(pc, menor)

        self.lbl_camelot.configure(text=cod, bg=cor, fg=texto_sobre(cor))
        self.var_tom.set(f"{nome_pt}  ({sigla})")
        margem = corr0 - corr1
        if corr0 >= 0.80 and margem >= 0.05:
            conf = "Alta"
        elif corr0 >= 0.65:
            conf = "Média"
        else:
            conf = "Baixa"
        self.var_conf_tom.set(f"Confiança: {conf}  (correlação {corr0:.2f})".replace(".", SEP_DECIMAL))

        self._desenhar_roda(cod)
        self.var_comp.set("Combinam na mixagem: " + ", ".join(sorted(compativeis(cod), key=lambda c: (int(c[:-1]), c[-1]))))
        self._desenhar_chroma(res["chroma"], pc, cor)

        for chip, (c, p, m) in zip(self.chips, res["ranking"][1:4]):
            cd = codigo_camelot(p, m)
            cr = cor_camelot(cd)
            n_pt, sg = nome_tom(p, m)
            chip.configure(text=f"{cd}   {n_pt} ({sg})   ·   {c:.2f}".replace(".", SEP_DECIMAL),
                           bg=cr, fg=texto_sobre(cr))
        self.var_status.set(f"Espectro e tom prontos: {cod} — {nome_pt}.")

    # ------------------------------- desenho ------------------------------- #
    def _agendar_redesenho(self, _evento=None):
        if self._redesenho_id:
            self.after_cancel(self._redesenho_id)
        self._redesenho_id = self.after(80, self._desenhar_espectro)

    def _desenhar_espectro(self, titulo=None):
        if titulo is not None:
            self._titulo_spec = titulo
        cv = self.cv_spec
        cv.delete("all")
        W, H = cv.winfo_width(), cv.winfo_height()
        if W < 120 or H < 120:
            return
        if self.spec is None:
            cv.create_text(W / 2, H / 2, fill="#9a9aa5", font=("Segoe UI", 11), justify="center",
                           text="O espectrograma aparece aqui.\nEscolha um arquivo e clique em\n\"Analisar espectro e tom\".")
            return

        import numpy as np

        ml, mr, mt, mb = 56, 12, 24, 28
        w, h = W - ml - mr, H - mt - mb
        if w < 50 or h < 50:
            return
        ppm = renderizar_ppm(self.spec, w, h)
        self.img_spec = tk.PhotoImage(width=w, height=h, data=ppm, format="PPM")  # manter referência!
        cv.create_image(ml, mt, image=self.img_spec, anchor="nw")
        cv.create_text(ml, 4, anchor="nw", fill="#d8d8e0", font=("Segoe UI", 9, "bold"),
                       text=getattr(self, "_titulo_spec", ""))

        # eixo de frequência (escala mel)
        n_mels = len(self.spec_mel)
        for f, rot in ((100, "100"), (250, "250"), (500, "500"), (1000, "1k"),
                       (2000, "2k"), (4000, "4k"), (8000, "8k")):
            i = int(np.argmin(np.abs(self.spec_mel - f)))
            y = mt + h - (i / (n_mels - 1)) * h
            cv.create_line(ml, y, ml + w, y, fill="#3a3a48", dash=(2, 6))
            cv.create_text(ml - 6, y, text=f"{rot} Hz", anchor="e", fill="#c8c8d0", font=("Segoe UI", 8))

        # eixo de tempo
        passo = next((p for p in (5, 10, 15, 30, 60, 120, 300, 600) if self.spec_dur / p <= max(2, w / 80)), 600)
        t = 0
        while t <= self.spec_dur:
            x = ml + (t / self.spec_dur) * w
            cv.create_line(x, mt + h, x, mt + h + 4, fill="#c8c8d0")
            cv.create_text(x, mt + h + 6, text=f"{int(t) // 60}:{int(t) % 60:02d}", anchor="n",
                           fill="#c8c8d0", font=("Segoe UI", 8))
            t += passo

    def _desenhar_roda(self, destaque=None):
        cv = self.cv_roda
        cv.delete("all")
        cx = cy = 130
        R, r, r0 = 124, 84, 46
        comp = compativeis(destaque) if destaque else set()

        def setor(r_in, r_out, theta):
            pts = []
            angulos = [theta - 15 + 30 * k / 8 for k in range(9)]
            for a in angulos:
                pts.append((cx + r_out * math.sin(math.radians(a)), cy - r_out * math.cos(math.radians(a))))
            for a in reversed(angulos):
                pts.append((cx + r_in * math.sin(math.radians(a)), cy - r_in * math.cos(math.radians(a))))
            return [c for p in pts for c in p]

        def desenhar(cod, contorno, largura):
            num, letra = int(cod[:-1]), cod[-1]
            theta = (num % 12) * 30
            r_in, r_out = (r, R) if letra == "B" else (r0, r)
            cor = cor_camelot(cod)
            if destaque and cod != destaque and cod not in comp:
                cor = misturar(cor, 0.65)
            cv.create_polygon(setor(r_in, r_out, theta), fill=cor, outline=contorno, width=largura)
            rr = (r_in + r_out) / 2
            cv.create_text(cx + rr * math.sin(math.radians(theta)), cy - rr * math.cos(math.radians(theta)),
                           text=cod, fill=texto_sobre(cor), font=("Segoe UI", 8, "bold"))

        todos = [f"{n}{l}" for n in range(1, 13) for l in ("B", "A")]
        for cod in todos:
            if cod != destaque and cod not in comp:
                desenhar(cod, "#ffffff", 1)
        for cod in comp:
            desenhar(cod, "#444444", 2)
        if destaque:
            desenhar(destaque, "#000000", 3)
        cv.create_text(cx, cy, text="B = maior\nA = menor", fill="#666666", font=("Segoe UI", 8), justify="center")

    def _desenhar_chroma(self, chroma=None, tonica=None, cor="#888888"):
        cv = self.cv_chroma
        cv.delete("all")
        if chroma is None:
            return
        W, H = 270, 90
        maior = float(max(chroma)) or 1.0
        larg = W / 12
        for i, v in enumerate(chroma):
            altura = (float(v) / maior) * (H - 26)
            x0, x1 = i * larg + 3, (i + 1) * larg - 3
            cv.create_rectangle(x0, H - 16 - altura, x1, H - 16, fill=cor if i == tonica else "#9aa0a6", outline="")
            cv.create_text((x0 + x1) / 2, H - 7, text=NOTAS_BARRAS[i], font=("Segoe UI", 7), fill="#333333")

    # --------------------------- estilos / faixa --------------------------- #
    def _aplicar_estilo(self, _evento=None):
        faixa = ESTILOS.get(self.var_estilo.get())
        if faixa:
            self._aplicando_estilo = True
            self.var_min.set(faixa[0])
            self.var_max.set(faixa[1])
            self._aplicando_estilo = False

    def _faixa_editada(self, *_):
        if not self._aplicando_estilo:
            self.var_estilo.set("Personalizado")

    # ------------------------------ helpers -------------------------------- #
    def _travar(self, ocupado):
        self.running = ocupado
        normal = "disabled" if ocupado else "normal"
        for b in (self.btn_pasta, self.btn_arqs, self.btn_analisar, self.btn_renomear):
            b.configure(state=normal)
        self.btn_cancelar.configure(state="normal" if ocupado else "disabled")
        self.btn_desfazer.configure(state="disabled" if ocupado or not self.undo_stack else "normal")

    def _linha(self, iid, arquivo=None, pasta=None, bpm=None, conf=None, novo=None, status=None, tag=None):
        vals = list(self.tree.item(iid, "values"))
        for i, v in enumerate((arquivo, pasta, bpm, conf, novo, status)):
            if v is not None:
                vals[i] = v
        self.tree.item(iid, values=vals)
        if tag:
            self.tree.item(iid, tags=(tag,))

    def _nome_novo(self, iid):
        it = self.items[iid]
        return f"{fmt_bpm(it['bpm'])} - {it['path'].name}"

    def _marcar_pronto(self, iid, bpm, conf, status="Pronto"):
        self.items[iid].update(bpm=bpm, state="ready")
        tag = "low" if conf == "Baixa" else "ready"
        self._linha(iid, bpm=fmt_bpm(bpm), conf=conf, novo=self._nome_novo(iid), status=status, tag=tag)

    # ------------------------------ adicionar ------------------------------ #
    def adicionar_pasta(self):
        pasta = filedialog.askdirectory(title="Escolha a pasta com as músicas")
        if not pasta:
            return
        p = Path(pasta)
        it = p.rglob("*") if self.var_sub.get() else p.glob("*")
        self._adicionar([a for a in sorted(it) if a.is_file() and a.suffix.lower() in EXTENSOES])

    def adicionar_arquivos(self):
        arqs = filedialog.askopenfilenames(
            title="Escolha as músicas",
            filetypes=[("Áudio", "*.mp3 *.flac *.wav"), ("Todos os arquivos", "*.*")],
        )
        self._adicionar([Path(a) for a in arqs])

    def _adicionar(self, caminhos):
        novos = 0
        for p in caminhos:
            if p in self.paths_set:
                continue
            iid = self.tree.insert("", "end", values=(p.name, str(p.parent), "", "", "", "Aguardando"))
            self.items[iid] = {"path": p, "bpm": None, "state": "wait"}
            self.paths_set.add(p)
            if PADRAO_JA_RENOMEADO.match(p.name):
                self.items[iid]["state"] = "skip"
                self._linha(iid, status="Já tem BPM", tag="skip")
            novos += 1
        self.var_status.set(f"{novos} arquivo(s) adicionado(s). Total na lista: {len(self.items)}.")

    # --------------------------- gerenciar lista --------------------------- #
    def selecionar_todos(self):
        self.tree.selection_set(self.tree.get_children())

    def remover_selecionados(self):
        if self.running:
            return
        for iid in self.tree.selection():
            self.paths_set.discard(self.items[iid]["path"])
            del self.items[iid]
            self.tree.delete(iid)
        self.var_status.set(f"Total na lista: {len(self.items)}.")

    def limpar_lista(self):
        if self.running:
            return
        self.tree.delete(*self.tree.get_children())
        self.items.clear()
        self.paths_set.clear()
        self.progresso["value"] = 0
        self.var_status.set("Lista limpa.")

    def editar_bpm(self, _evento):
        sel = self.tree.selection()
        if self.running or len(sel) != 1:
            return
        iid = sel[0]
        it = self.items[iid]
        if it["state"] not in ("ready", "error", "wait"):
            return
        txt = simpledialog.askstring(
            "Corrigir BPM",
            f"BPM de:\n{it['path'].name}\n\nDigite o valor (ex.: 148,50):",
            parent=self,
            initialvalue=fmt_bpm(it["bpm"]) if it["bpm"] else "148,00",
        )
        if not txt:
            return
        try:
            valor = round(float(txt.strip().replace(",", ".")), 2)
        except ValueError:
            messagebox.showwarning("BPM inválido", "Digite um número, por exemplo 148,50.")
            return
        if not 30 <= valor <= 800:
            messagebox.showwarning("BPM inválido", "O BPM deve estar entre 30 e 800.")
            return
        self._marcar_pronto(iid, valor, "Manual", status="Pronto (manual)")

    def multiplicar(self, fator):
        if self.running:
            return
        for iid in self.tree.selection():
            it = self.items[iid]
            if it["state"] != "ready" or it["bpm"] is None:
                continue
            novo = round(it["bpm"] * fator, 2)
            if 30 <= novo <= 800:
                self._marcar_pronto(iid, novo, "Manual", status="Pronto (ajustado)")

    # ------------------------------- analisar ------------------------------ #
    def analisar(self):
        try:
            minimo, maximo = float(self.var_min.get()), float(self.var_max.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Faixa de BPM", "Informe números válidos para a faixa de BPM.")
            return
        if minimo < 30 or maximo <= minimo * 1.02:
            messagebox.showwarning(
                "Faixa de BPM",
                "Informe uma faixa válida (mínimo a partir de 30 e máximo maior que o mínimo).",
            )
            return

        sel = [i for i in self.tree.selection() if self.items[i]["state"] in ("wait", "ready", "error")]
        alvo = sel or [i for i, it in self.items.items() if it["state"] in ("wait", "error")]
        if not alvo:
            messagebox.showinfo("Analisar", "Não há músicas pendentes para analisar.")
            return

        self.cancel.clear()
        self.progresso.configure(maximum=len(alvo), value=0)
        self._travar(True)
        trabalho = [(i, self.items[i]["path"]) for i in alvo]
        modo = MODOS.get(self.var_modo.get(), "preciso")
        encaixar = bool(self.var_snap.get())
        threading.Thread(target=self._worker, args=(trabalho, minimo, maximo, modo, encaixar),
                         daemon=True).start()

    def _worker(self, trabalho, minimo, maximo, modo, encaixar):
        self.q.put(("msg", "Carregando o motor de análise (na primeira vez pode demorar)..."))
        try:
            import librosa  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.q.put(("fatal", "Faltam bibliotecas.\nInstale com:\n\npip install librosa numpy soundfile"))
            return
        total = len(trabalho)
        for n, (iid, path) in enumerate(trabalho, 1):
            if self.cancel.is_set():
                break
            self.q.put(("status", iid, "Analisando..."))
            try:
                bpm, conf, aviso = detectar_bpm(path, minimo, maximo, modo, encaixar)
                self.q.put(("result", iid, bpm, conf, aviso))
            except Exception as e:  # arquivo corrompido, sem ffmpeg, muito curto etc.
                self.q.put(("error", iid, str(e)))
            self.q.put(("progress", n, total))
        self.q.put(("done", self.cancel.is_set()))

    def cancelar(self):
        self.cancel.set()
        self.var_status.set("Cancelando após a música atual...")

    def _processar_fila(self):
        try:
            while True:
                msg = self.q.get_nowait()
                tipo = msg[0]
                if tipo == "msg":
                    self.var_status.set(msg[1])
                elif tipo == "status" and msg[1] in self.items:
                    self._linha(msg[1], status=msg[2])
                elif tipo == "result" and msg[1] in self.items:
                    status = "Verifique a oitava" if msg[4] else "Pronto"
                    self._marcar_pronto(msg[1], msg[2], msg[3], status=status)
                elif tipo == "error" and msg[1] in self.items:
                    self.items[msg[1]]["state"] = "error"
                    self._linha(msg[1], status="Erro", tag="error")
                elif tipo == "progress":
                    self.progresso["value"] = msg[1]
                    self.var_status.set(f"Analisando {msg[1]} de {msg[2]}...")
                elif tipo == "fatal":
                    self._travar(False)
                    messagebox.showerror("Bibliotecas ausentes", msg[1])
                    self.var_status.set("Erro: bibliotecas ausentes.")
                elif tipo == "done":
                    self._travar(False)
                    prontos = sum(1 for it in self.items.values() if it["state"] == "ready")
                    txt = "Análise cancelada." if msg[1] else "Análise concluída."
                    self.var_status.set(
                        f"{txt} {prontos} música(s) pronta(s). Confira as de confiança \"Baixa\" "
                        "(em laranja) antes de renomear."
                    )
                # --- aba de espectro e tom ---
                elif tipo == "esp_status":
                    self.var_status.set(msg[1])
                elif tipo == "esp_ok":
                    self._esp_finalizar()
                    self._mostrar_resultado_esp(msg[1], msg[2])
                elif tipo == "esp_err":
                    self._esp_finalizar()
                    self.var_status.set("Não foi possível analisar o espectro e o tom.")
                    messagebox.showerror("Espectro e Tom", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._processar_fila)

    # ------------------------------- renomear ------------------------------ #
    def renomear(self):
        sel = [i for i in self.tree.selection() if self.items[i]["state"] == "ready"]
        alvo = sel or [i for i, it in self.items.items() if it["state"] == "ready"]
        if not alvo:
            messagebox.showinfo("Renomear", "Nenhuma música analisada e pronta para renomear.\n"
                                            "Clique em \"Analisar BPM\" primeiro.")
            return
        if not messagebox.askyesno("Confirmar", f"Renomear {len(alvo)} arquivo(s) agora?\n\n"
                                                "Você poderá desfazer depois, se precisar."):
            return

        lote, erros = [], 0
        for iid in alvo:
            it = self.items[iid]
            antigo = it["path"]
            novo = antigo.with_name(self._nome_novo(iid))
            try:
                if novo.exists():
                    raise FileExistsError("já existe um arquivo com esse nome")
                antigo.rename(novo)
            except Exception:
                erros += 1
                self._linha(iid, status="Falhou", tag="error")
                continue
            self.paths_set.discard(antigo)
            self.paths_set.add(novo)
            it["path"], it["state"] = novo, "done"
            if self.esp_caminho == antigo:      # mantém a aba de espectro apontando para o arquivo certo
                self.esp_caminho = novo
                self.var_esp_arq.set(f"Arquivo: {novo.name}")
            self._linha(iid, arquivo=novo.name, novo="", status="Renomeado ✔", tag="done")
            lote.append((iid, novo, antigo))

        if lote:
            self.undo_stack.append(lote)
            self.btn_desfazer.configure(state="normal")
        msg = f"{len(lote)} arquivo(s) renomeado(s)."
        if erros:
            msg += f" {erros} falharam (nome já existe ou arquivo em uso)."
        self.var_status.set(msg)
        if erros:
            messagebox.showwarning("Atenção", msg)

    def desfazer(self):
        if not self.undo_stack:
            return
        lote = self.undo_stack.pop()
        revertidos = 0
        for iid, novo, antigo in reversed(lote):
            try:
                if not (novo.exists() and not antigo.exists()):
                    continue
                novo.rename(antigo)
            except Exception:
                continue
            revertidos += 1
            if self.esp_caminho == novo:
                self.esp_caminho = antigo
                self.var_esp_arq.set(f"Arquivo: {antigo.name}")
            if iid in self.items:
                it = self.items[iid]
                self.paths_set.discard(novo)
                self.paths_set.add(antigo)
                it["path"] = antigo
                self._marcar_pronto(iid, it["bpm"], "—")
        self.btn_desfazer.configure(state="normal" if self.undo_stack else "disabled")
        self.var_status.set(f"Desfeito: {revertidos} arquivo(s) voltaram ao nome original.")

    def _fechar(self):
        self.cancel.set()
        self.destroy()


if __name__ == "__main__":
    try:  # texto nítido em telas de alta resolução no Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
