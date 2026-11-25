"""Mapa de canales virtuales 5.1 -> parlantes reales según su posición en el cuarto (VBAP 2D)."""
from __future__ import annotations

import math

import numpy as np

# Orden de canales virtuales (coincide con el orden 5.1 de ffmpeg: FL FR FC LFE BL BR)
CHANNELS = ["L", "R", "C", "LFE", "Ls", "Rs"]
# Azimut ideal de cada canal (grados, 0 = frente, positivo = derecha). ITU-R BS.775.
AZIMUTH = {"L": -30.0, "R": 30.0, "C": 0.0, "Ls": -110.0, "Rs": 110.0}


def speaker_azimuth(spk: dict, listener: dict) -> float:
    """Ángulo del parlante visto desde el oyente. Coordenadas en metros: x a la derecha, y hacia la pantalla."""
    dx = spk["x"] - listener["x"]
    dy = spk["y"] - listener["y"]
    return math.degrees(math.atan2(dx, dy))


def speaker_distance(spk: dict, listener: dict) -> float:
    return math.hypot(spk["x"] - listener["x"], spk["y"] - listener["y"])


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _pair_gains(src: float, a1: float, a2: float):
    """Ganancias VBAP 2D para una fuente entre dos parlantes. None si queda fuera del arco o el par es degenerado."""
    def vec(a):
        r = math.radians(a)
        return np.array([math.sin(r), math.cos(r)])
    base = np.column_stack([vec(a1), vec(a2)])
    if abs(np.linalg.det(base)) < 1e-6:
        return None
    g = np.linalg.solve(base, vec(src))
    if g.min() < -1e-6:
        return None
    g = np.clip(g, 0, None)
    n = np.linalg.norm(g)
    return g / n if n > 0 else None


def panning_gains(src_az: float, spk_az: list) -> np.ndarray:
    """Reparte una fuente entre los dos parlantes vecinos del arco (circular) donde cae.

    Arco < 180°: VBAP (ley de la tangente, la imagen queda entre los dos). Arco ≥ 180° (p. ej. el frente
    entre dos laterales): VBAP no sirve (los vectores se oponen), así que se reparte por fracción de ángulo
    con potencia constante. Así es continuo en todo el círculo y nunca usa un parlante del lado opuesto
    (antes, un sonido adelante-izquierda salía en parte por el parlante de atrás)."""
    n = len(spk_az)
    g = np.zeros(n)
    if n == 1:
        g[0] = 1.0
        return g
    order = sorted(range(n), key=lambda i: spk_az[i] % 360.0)
    src = src_az % 360.0
    for k in range(n):
        i, j = order[k], order[(k + 1) % n]
        a, span = spk_az[i] % 360.0, (spk_az[j] - spk_az[i]) % 360.0
        if span == 0.0:
            span = 360.0                       # dos parlantes en el mismo ángulo (o uno solo efectivo)
        t = (src - a) % 360.0
        if t > span + 1e-9:
            continue
        if span < 180.0 - 1e-6:
            pg = _pair_gains(src_az, spk_az[i], spk_az[j])
            if pg is not None:
                g[i], g[j] = pg
                return g
        f = t / span
        g[i], g[j] = math.cos(f * math.pi / 2), math.sin(f * math.pi / 2)
        return g
    g[int(np.argmin([abs(_wrap(src_az - a)) for a in spk_az]))] = 1.0
    return g


def build_matrix(speakers: list, listener: dict, width: float = 30.0) -> np.ndarray:
    """Devuelve matriz (6 canales virtuales x N parlantes). `width`: ángulo de L/R (30° = estándar;
    más abre el escenario hacia los costados)."""
    az = [speaker_azimuth(s, listener) for s in speakers]
    target = {**AZIMUTH, "L": -float(width), "R": float(width)}
    n = len(speakers)
    m = np.zeros((len(CHANNELS), n))
    for ci, ch in enumerate(CHANNELS):
        if ch == "LFE":
            m[ci, :] = 0.7 / math.sqrt(n)
            continue
        m[ci, :] = panning_gains(target[ch], az)
    # Los canales traseros sin parlantes detrás llegan más suaves para no tapar el frente.
    for ch in ("Ls", "Rs"):
        ci = CHANNELS.index(ch)
        if not any(abs(a) > 90 for a in az):
            m[ci, :] *= 0.6
    return m.astype(np.float32)


def distance_alignment(speakers: list, listener: dict, sr: int):
    """Retraso (muestras) y ganancia para igualar parlantes a distintas distancias (si no hay calibración con mic)."""
    d = np.array([speaker_distance(s, listener) for s in speakers])
    delays = ((d.max() - d) / 343.0 * sr).round().astype(int)
    gains_db = 20 * np.log10(np.maximum(d, 0.3) / max(d.max(), 0.3))
    return delays, gains_db


def describe(speakers: list, listener: dict) -> str:
    m = build_matrix(speakers, listener)
    lines = []
    head = "canal  " + "  ".join(f"{s['name'][:10]:>10}" for s in speakers)
    lines.append(head)
    for ci, ch in enumerate(CHANNELS):
        lines.append(f"{ch:<6} " + "  ".join(f"{v:>10.2f}" for v in m[ci]))
    lines.append("")
    for s in speakers:
        lines.append(f"{s['name']}: azimut {speaker_azimuth(s, listener):6.1f}°, "
                     f"distancia {speaker_distance(s, listener):.2f} m")
    return "\n".join(lines)
