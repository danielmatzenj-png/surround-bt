"""Buffer circular mono con lectura remuestreada (corrige deriva de reloj entre parlantes)."""
from __future__ import annotations

import threading

import numpy as np

HALF = 8            # puntos a cada lado para la interpolación sinc
CUTOFF = 0.94       # deja pasar hasta ~22,5 kHz a 48 kHz (antialias leve)
BETA = 8.0          # ventana Kaiser
I0B = float(np.i0(BETA))
SINC = True        # False = interpolación lineal (solo para comparar)


class Fifo:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.r = 0          # índice absoluto de lectura (entero)
        self.frac = 0.0     # fase fraccional de lectura
        self.w = 0          # índice absoluto de escritura
        self.pad = 0        # ceros pendientes a emitir sin consumir (para aumentar retraso)
        self.underruns = 0
        self.lock = threading.Lock()

    def fill(self) -> float:
        return (self.w - self.r - self.frac) + self.pad

    def write(self, x: np.ndarray) -> None:
        n = len(x)
        with self.lock:
            if n >= self.cap:
                x = x[-self.cap:]
                n = self.cap
            overflow = (self.w - self.r) + n - self.cap
            if overflow > 0:
                self.r += overflow
            i = self.w % self.cap
            first = min(n, self.cap - i)
            self.buf[i:i + first] = x[:first]
            if first < n:
                self.buf[:n - first] = x[first:]
            self.w += n

    def drop(self, n: int) -> None:
        with self.lock:
            k = min(self.pad, int(n))          # primero lo que falta por rellenar
            self.pad -= k
            n = int(min(n - k, self.w - self.r))
            self.r += max(n, 0)

    def add_pad(self, n: int) -> None:
        with self.lock:
            self.pad += int(max(n, 0))

    def read(self, nout: int, ratio: float) -> np.ndarray:
        out = np.zeros(nout, dtype=np.float32)
        with self.lock:
            start = 0
            if self.pad:
                k = min(self.pad, nout)
                self.pad -= k
                start = k
            m = nout - start
            if m <= 0:
                return out
            pos = self.frac + ratio * np.arange(m)
            avail = self.w - self.r
            need = int(pos[-1]) + HALF + 1
            if avail < need:
                # Subdesbordamiento: entrega lo que hay, el resto silencio.
                self.underruns += 1
                m = max(0, min(m, int((avail - HALF - 1 - self.frac) / ratio)))
                pos = pos[:m]
                if m == 0:
                    return out
            i0 = np.floor(pos).astype(np.int64)
            t = pos - i0
            if not SINC:
                a = self.buf[(self.r + i0) % self.cap]
                b = self.buf[(self.r + i0 + 1) % self.cap]
                out[start:start + m] = a + (b - a) * t
                adv = self.frac + ratio * m
                whole = int(np.floor(adv))
                self.r += whole
                self.frac = adv - whole
                return out
            # Sinc con ventana Kaiser, 16 puntos: limpio hasta ~20 kHz (la lineal ensuciaba los agudos).
            k = np.arange(-HALF + 1, HALF + 1)                 # -7..8
            x = t[:, None] - k[None, :]                        # distancia a cada muestra vecina
            wts = np.sinc(x * CUTOFF) * CUTOFF * np.i0(BETA * np.sqrt(np.clip(1 - (x / HALF) ** 2, 0, 1))) / I0B
            idx = (self.r + i0[:, None] + k[None, :]) % self.cap
            out[start:start + m] = (self.buf[idx] * wts).sum(axis=1)
            adv = self.frac + ratio * m
            whole = int(np.floor(adv))
            self.r += whole
            self.frac = adv - whole
        return out
