"""Esfera de sonido: ubica sonidos-objeto por dirección (azimut), altura (elevación) y distancia.

Con parlantes a una sola altura no hay forma física de poner un sonido arriba; lo que sí funciona es darle al
oído las señales con que juzga cada cosa:
- Dirección horizontal: reparto entre los dos parlantes vecinos (`render.panning_gains`), con rampa por
  bloque para que moverlo no haga clics.
- Delante / detrás: bandas direccionales de Blauert. Lo de adelante se refuerza en 3–4 kHz; lo de atrás
  pierde brillo (>5 kHz) y gana algo en ~1 kHz.
- Altura: el pabellón de la oreja marca lo de arriba con más energía en ~8 kHz y una muesca que sube de
  frecuencia con la elevación (~6 → 10 kHz). Abajo: menos agudos.
- Distancia: nivel 1/d, el aire se come los agudos con la distancia, y más reverberación difusa (la
  proporción directo/reverberado es la señal de distancia más fuerte).
- Si los parlantes tienen alturas distintas (`z`), además se reparte entre los de abajo y los de arriba.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt

from . import render
from .sources_base import SR

EAR_Z = 1.1          # m: oído sentado
REF_DIST = 1.0       # m: a esta distancia el objeto suena a su nivel nominal


def _peak(f0: float, gain_db: float, q: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / SR
    al = math.sin(w) / (2 * q)
    b = [1 + al * a, -2 * math.cos(w), 1 - al * a]
    d = [1 + al / a, -2 * math.cos(w), 1 - al / a]
    return np.array([b[0] / d[0], b[1] / d[0], b[2] / d[0], 1.0, d[1] / d[0], d[2] / d[0]])


def _shelf_hi(f0: float, gain_db: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / SR
    al = math.sin(w) / 2 * math.sqrt(2)
    c, sa = math.cos(w), 2 * math.sqrt(a) * al
    b0 = a * ((a + 1) + (a - 1) * c + sa)
    b1 = -2 * a * ((a - 1) + (a + 1) * c)
    b2 = a * ((a + 1) + (a - 1) * c - sa)
    a0 = (a + 1) - (a - 1) * c + sa
    a1 = 2 * ((a - 1) - (a + 1) * c)
    a2 = (a + 1) - (a - 1) * c - sa
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def _air_lp(dist: float) -> np.ndarray:
    """Pasabajos suave (1 polo en forma de biquad) que baja con la distancia: ~18 kHz a 1 m, ~6 kHz a 12 m."""
    fc = float(np.clip(18000.0 / (1.0 + 0.18 * max(0.0, dist - 1.0)), 5000.0, 18000.0))
    x = math.exp(-2 * math.pi * fc / SR)
    return np.array([1 - x, 0.0, 0.0, 1.0, -x, 0.0])


def cue_filters(az: float, el: float, dist: float) -> np.ndarray:
    """Cadena (sos) de señales de dirección, altura y distancia para un objeto."""
    front = max(0.0, math.cos(math.radians(az)))          # 1 adelante, 0 a los costados y atrás
    back = max(0.0, -math.cos(math.radians(az)))
    up = max(0.0, math.sin(math.radians(el)))
    down = max(0.0, -math.sin(math.radians(el)))
    sos = [
        _peak(3500.0, 3.0 * front, 1.2),                   # adelante: presencia
        _peak(1000.0, 2.0 * back, 1.0),                    # atrás: banda de ~1 kHz
        _shelf_hi(5000.0, -4.0 * back - 4.0 * down),       # atrás / abajo: menos brillo
        _peak(8000.0, 6.0 * up, 1.4),                      # arriba: banda de 8 kHz
        _peak(6500.0 + 5500.0 * up, -7.0 * up, 6.0),       # arriba: muesca del pabellón (sube con la altura)
        _air_lp(dist),
    ]
    return np.array(sos)


def speaker_elevation(spk: dict, listener: dict) -> float:
    z = spk.get("z")
    if z is None:
        return 0.0
    d = max(0.3, render.speaker_distance(spk, listener))
    return math.degrees(math.atan2(float(z) - EAR_Z, d))


def gains(az: float, el: float, spk_az: list, spk_el: list) -> np.ndarray:
    """Reparto entre parlantes. Con dos alturas de parlantes, se reparte por capas (abajo/arriba) y dentro
    de cada capa por azimut; si están todos a la misma altura, solo por azimut."""
    spk_el = np.asarray(spk_el, dtype=float)
    if len(spk_az) < 2 or np.ptp(spk_el) < 15.0:
        return render.panning_gains(az, spk_az)
    split = float(np.median(spk_el))
    lo = [i for i, e in enumerate(spk_el) if e <= split]
    hi = [i for i, e in enumerate(spk_el) if e > split]
    if not lo or not hi:
        return render.panning_gains(az, spk_az)
    e_lo, e_hi = float(np.mean(spk_el[lo])), float(np.mean(spk_el[hi]))
    f = float(np.clip((el - e_lo) / max(1e-6, e_hi - e_lo), 0.0, 1.0))
    g = np.zeros(len(spk_az))
    for idx, w in ((lo, math.cos(f * math.pi / 2)), (hi, math.sin(f * math.pi / 2))):
        g[idx] += w * render.panning_gains(az, [spk_az[i] for i in idx])
    return g / (np.linalg.norm(g) + 1e-12)


class Reverb:
    """Reverberación difusa y barata: por parlante, 3 peines en paralelo + 2 pasatodos (Schroeder) con
    tiempos distintos en cada uno, así sale decorrelacionada y envuelve sin ubicarse en ningún parlante."""

    def __init__(self, n_spk: int, rt: float = 0.45):
        self.n = n_spk
        rng = np.random.default_rng(7)
        self.combs, self.aps = [], []
        for _ in range(n_spk):
            ds = (rng.uniform(0.023, 0.041, 3) * SR).astype(int)
            self.combs.append([(d, 10 ** (-3 * d / SR / rt), np.zeros(d, np.float32), 0) for d in ds])
            aps = (rng.uniform(0.004, 0.009, 2) * SR).astype(int)
            self.aps.append([(d, 0.6, np.zeros(d, np.float32), 0) for d in aps])

    def _line(self, x: np.ndarray, state: list, comb: bool) -> tuple:
        out = np.zeros_like(x)
        new = []
        for d, g, buf, p in state:
            y = np.empty_like(x)
            for k in range(0, len(x), d):                  # bloques ≤ d: la recursión cae fuera del bloque
                m = min(d, len(x) - k)
                idx = (p + np.arange(m)) % d
                past = buf[idx]
                if comb:
                    y[k:k + m] = past
                    buf[idx] = x[k:k + m] + g * past
                else:
                    v = x[k:k + m] + g * past
                    y[k:k + m] = past - g * v
                    buf[idx] = v
                p = (p + m) % d
            new.append((d, g, buf, p))
            out = out + y if comb else y
            if not comb:
                x = y
        return (out / len(state) if comb else out), new

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), self.n), np.float32)
        for i in range(self.n):
            y, self.combs[i] = self._line(x, self.combs[i], True)
            y, self.aps[i] = self._line(y, self.aps[i], False)
            out[:, i] = y
        return out


class ObjectRenderer:
    """Un objeto en la esfera → señal para cada parlante. Mantiene filtros y ganancias entre bloques."""

    def __init__(self, n_spk: int):
        self.n = n_spk
        self.g = None
        self.key = None
        self.sos = None
        self.zi = None
        self.verb = Reverb(n_spk)

    def __call__(self, sig: np.ndarray, az: float, el: float, dist: float, spk_az: list, spk_el: list) -> np.ndarray:
        n = len(sig)
        dist = max(0.3, float(dist))
        # Filtros: se recalculan solo si el objeto se movió lo suficiente (evita trabajo y zumbido de
        # coeficientes). El estado se conserva: misma estructura de 6 secciones.
        key = (round(az / 2), round(el / 2), round(dist * 10))
        if key != self.key:
            self.sos = cue_filters(az, el, dist)
            if self.zi is None:
                self.zi = np.zeros((self.sos.shape[0], 2))
            self.key = key
        dry, self.zi = sosfilt(self.sos, sig.astype(np.float64), zi=self.zi)
        level = REF_DIST / max(0.5, dist)
        g_new = gains(az, el, spk_az, spk_el) * level
        g_old = g_new if self.g is None else self.g
        ramp = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out = (dry[:, None] * (g_old + (g_new - g_old) * ramp)).astype(np.float32)
        self.g = g_new
        # Reverberación: crece con la distancia (lejos = más sala, menos directo)
        wet = float(np.clip((dist - 0.7) / 6.0, 0.05, 0.6)) * level
        out += self.verb(sig.astype(np.float32) * wet)
        return out
