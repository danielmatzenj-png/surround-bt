"""Estéreo -> 5.1 por separación primario/ambiente en el dominio de frecuencia (STFT).

En cada banda y cuadro:
- La coherencia entre L y R separa lo "primario" (fuentes directas: voces, instrumentos) del
  "ambiente" (reverberación, público, efectos difusos).
- Lo primario que está igual en L y R (paneado al centro: casi siempre los diálogos) va al canal C.
- El ambiente va a los traseros, retrasado y suavizado para que envuelva sin distraer.
Así el centro estabiliza las voces y los traseros no repiten lo de adelante.
"""
from __future__ import annotations

import numpy as np

from .sources_base import SR

N = 1024          # cuadro (21 ms)
H = N // 2        # salto
_WIN = np.sqrt(np.hanning(N + 1)[:N]).astype(np.float32)   # sqrt-Hann: análisis+síntesis = 1


class PAUpmixer:
    def __init__(self, center: float = 1.0, surround: float = 0.85, front_ambience: float = 0.5,
                 smoothing: float = 0.8):
        self.center, self.surround, self.front_amb = center, surround, front_ambience
        self.alpha = smoothing
        nb = N // 2 + 1
        self.pll = np.full(nb, 1e-9)
        self.prr = np.full(nb, 1e-9)
        self.plr = np.zeros(nb, complex)
        self.prev = np.zeros((H, 2), np.float32)
        self.tail = np.zeros((H, 5), np.float32)      # L R C SL SR
        self.inq = np.zeros((0, 2), np.float32)
        self.outq = np.zeros((H, 5), np.float32)      # pre-relleno = latencia fija de H muestras
        f = np.fft.rfftfreq(N, 1 / SR)
        self.sur_lp = np.clip((9000 - f) / 3000, 0, 1)  # traseros sin brillo excesivo (>7 kHz)
        # En graves los cuadros de 21 ms no separan bien (cada banda mide ~47 Hz) y la separación
        # «tiembla» de cuadro a cuadro: suena como estática. Ahí el sonido pasa intacto (transición
        # suave de 150 a 300 Hz); los graves los ubica después el gestor de graves.
        self.sep = np.clip((f - 150) / 150, 0, 1)
        # retrasos de los traseros (efecto Haas: suenan "detrás", no como eco)
        self.set_rear_delay(True, True)

    def set_rear_delay(self, left: bool, right: bool):
        """Retraso de los traseros solo donde hay parlantes propios atrás (si no, hace filtro peine)."""
        self.dl, self.dr = (int(0.012 * SR) if left else 0), (int(0.017 * SR) if right else 0)
        self.hist_l = np.zeros(self.dl, np.float32)
        self.hist_r = np.zeros(self.dr, np.float32)

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        L = np.fft.rfft(frame[:, 0] * _WIN)
        R = np.fft.rfft(frame[:, 1] * _WIN)
        a = self.alpha
        self.pll = a * self.pll + (1 - a) * (L.real ** 2 + L.imag ** 2)
        self.prr = a * self.prr + (1 - a) * (R.real ** 2 + R.imag ** 2)
        self.plr = a * self.plr + (1 - a) * (L * np.conj(R))
        # PCA por banda: el autovalor menor de la covarianza L/R es la energía de ambiente (igual en
        # ambos canales y sin correlación). Así un sonido paneado a un lado no se confunde con ambiente.
        half = 0.5 * (self.pll + self.prr)
        lam2 = np.maximum(half - np.sqrt((0.5 * (self.pll - self.prr)) ** 2 + np.abs(self.plr) ** 2), 0)
        amb_l = np.sqrt(np.clip(lam2 / (self.pll + 1e-12), 0, 1))
        amb_r = np.sqrt(np.clip(lam2 / (self.prr + 1e-12), 0, 1))
        sim = np.clip(2 * self.plr.real / (self.pll + self.prr + 1e-12), 0, 1)
        cmask = sim ** 2                               # qué tan "al centro" está lo primario
        amb_l, amb_r, cmask = amb_l * self.sep, amb_r * self.sep, cmask * self.sep
        PL, PR = L * (1 - amb_l), R * (1 - amb_r)
        AL, AR = L * amb_l, R * amb_r
        C = cmask * 0.5 * (PL + PR)
        out = np.empty((N, 5), np.float32)
        out[:, 0] = np.fft.irfft(PL - C + self.front_amb * AL, N)
        out[:, 1] = np.fft.irfft(PR - C + self.front_amb * AR, N)
        out[:, 2] = np.fft.irfft(C * np.sqrt(2) * self.center, N)
        out[:, 3] = np.fft.irfft(AL * self.sur_lp * self.surround, N)
        out[:, 4] = np.fft.irfft(-AR * self.sur_lp * self.surround, N)
        return out * _WIN[:, None]

    def __call__(self, st: np.ndarray) -> np.ndarray:
        n = len(st)
        self.inq = np.concatenate([self.inq, st[:, :2].astype(np.float32)])
        hops = []
        while len(self.inq) >= H:
            hop, self.inq = self.inq[:H], self.inq[H:]
            y = self._frame(np.concatenate([self.prev, hop]))
            self.prev = hop
            hops.append(self.tail + y[:H])
            self.tail = y[H:]
        if hops:
            self.outq = np.concatenate([self.outq] + hops)
        o, self.outq = self.outq[:n], self.outq[n:]
        if len(o) < n:
            o = np.pad(o, ((0, n - len(o)), (0, 0)))
        # retrasos de los traseros
        sl = np.concatenate([self.hist_l, o[:, 3]])
        sr = np.concatenate([self.hist_r, o[:, 4]])
        self.hist_l = sl[len(sl) - self.dl:]
        self.hist_r = sr[len(sr) - self.dr:]
        res = np.zeros((n, 6), np.float32)
        res[:, 0], res[:, 1], res[:, 2] = o[:, 0], o[:, 1], o[:, 2]
        res[:, 4], res[:, 5] = sl[:n], sr[:n]
        return res                                      # LFE en 0: los graves los reparte el gestor de graves
