"""Ecualización.

- Del usuario: 8 bandas (biquads RBJ), igual para todos los parlantes.
- Automática (igualación): curva por parlante en 1/3 de octava (25 bandas, 50 Hz–12,5 kHz),
  aplicada con un FIR de fase lineal. Todos los parlantes usan un FIR del mismo largo, así el
  retraso que agrega es idéntico y no rompe la sincronía.
"""
from __future__ import annotations

import numpy as np

from .sources import SR

# --- ecualizador del usuario ----------------------------------------------------------
BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
Q = 1.41           # ~1 octava por banda

PRESETS = {
    "flat":     [0, 0, 0, 0, 0, 0, 0, 0],
    "cinema":   [4, 3, 1, 0, 1, 1, 2, 2],
    "music":    [3, 2, 0, -1, 0, 1, 2, 3],
    "dialogue": [-3, -2, -1, 1, 3, 3, 1, 0],
    "night":    [-6, -4, -1, 1, 2, 2, 0, -2],
}

# --- igualación automática -------------------------------------------------------------
THIRD = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
         2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
_f = np.array(THIRD, dtype=float)
LOUD = np.where((_f >= 125) & (_f <= 8000))[0]     # rango con el que se compara volumen y tono
MAX_BOOST = 6.0    # dB: los parlantes pequeños se distorsionan si se empujan más
MAX_CUT = -15.0
BASS_FLAT_MAX = 6.0  # dB: lo más que se corrige la zona de graves del parlante que los lleva
MIN_SNR = 12.0     # dB sobre el ruido del cuarto para confiar en una banda
MIC_LOW_HZ = 140.0 # por debajo, el micrófono de la Mac no es confiable para subir graves
FIR_TAPS = 1025    # ~11 ms de retraso, igual para todos los parlantes


def _biquad(kind: str, f0: float, gain_db: float, q: float = Q) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    else:  # repisas (shelf) con pendiente S=1
        alpha = sw / 2 * np.sqrt(2)
        sa = 2 * np.sqrt(a) * alpha
        if kind == "low":
            b = [a * ((a + 1) - (a - 1) * cw + sa), 2 * a * ((a - 1) - (a + 1) * cw), a * ((a + 1) - (a - 1) * cw - sa)]
            den = [(a + 1) + (a - 1) * cw + sa, -2 * ((a - 1) + (a + 1) * cw), (a + 1) + (a - 1) * cw - sa]
        else:
            b = [a * ((a + 1) + (a - 1) * cw + sa), -2 * a * ((a - 1) + (a + 1) * cw), a * ((a + 1) + (a - 1) * cw - sa)]
            den = [(a + 1) - (a - 1) * cw + sa, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - sa]
    b = np.array(b) / den[0]
    den = np.array(den) / den[0]
    return np.concatenate([b, den])


def sos(gains_db) -> np.ndarray | None:
    """Cadena de secciones del ecualizador del usuario; None si todo está en 0 dB."""
    g = [float(x) for x in gains_db]
    if all(abs(x) < 0.05 for x in g):
        return None
    sections = []
    for i, (f, gd) in enumerate(zip(BANDS, g)):
        if abs(gd) < 0.05:
            continue
        kind = "low" if i == 0 else "high" if i == len(BANDS) - 1 else "peak"
        # Las repisas dan la mitad de su ganancia en la esquina: córrela media octava hacia adentro.
        fc = f * 1.41 if kind == "low" else f / 1.41 if kind == "high" else f
        sections.append(_biquad(kind, fc, gd))
    return np.array(sections)


def user_bands(cfg: dict) -> list:
    return [float(x) for x in cfg.get("eq", {}).get("bands", PRESETS["flat"])]


def auto_for(speaker_cfg: dict, cfg: dict) -> list:
    """Curva automática del parlante (25 valores); ceros si está apagada o es de una versión vieja."""
    a = speaker_cfg.get("eq_auto")
    if not cfg.get("eq_match", True) or not a or len(a) != len(THIRD):
        return [0.0] * len(THIRD)
    return [float(x) for x in a]


def fir(curve_db) -> np.ndarray:
    """FIR de fase lineal que sigue la curva (dB en THIRD). Fuera de 40 Hz–15 kHz vuelve a 0 dB."""
    nfft = 4096
    f = np.fft.rfftfreq(nfft, 1 / SR)
    lf = np.log2(np.maximum(f, 1.0))
    db = np.interp(lf, np.log2(_f), np.asarray(curve_db, dtype=float))
    fade = np.clip((f - 30) / 20, 0, 1) * np.clip((16000 - f) / 2000, 0, 1)
    mag = 10 ** (db * fade / 20)
    h = np.fft.irfft(mag, nfft)
    h = np.roll(h, FIR_TAPS // 2)[:FIR_TAPS] * np.hanning(FIR_TAPS)
    return h.astype(np.float32)


# --- medición ----------------------------------------------------------------------------
def band_levels(segment: np.ndarray, reference: np.ndarray) -> list:
    """Nivel (dB) del segmento grabado respecto a la señal emitida, por banda de 1/3 de octava."""
    n = 1 << int(np.ceil(np.log2(max(len(segment), len(reference)) * 2)))
    ps = np.abs(np.fft.rfft(segment, n)) ** 2
    pr = np.abs(np.fft.rfft(reference, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    k = 2 ** (1 / 6)
    out = []
    for fc in THIRD:
        m = (f >= fc / k) & (f < fc * k)
        out.append(float(10 * np.log10((ps[m].sum() + 1e-20) / (pr[m].sum() + 1e-20))))
    return out


def loudness(bands) -> float:
    """Volumen (dB) en 125 Hz–8 kHz, promediando energía (no decibeles)."""
    b = np.asarray(bands, dtype=float)[LOUD]
    return float(10 * np.log10(np.mean(10 ** (b / 10))))


def tone_spread(levels: list) -> tuple:
    """(máx, rms) de la diferencia de forma entre parlantes en 125 Hz–8 kHz, ya sin el volumen."""
    L = np.array([np.asarray(b, dtype=float) - loudness(b) for b in levels])[:, LOUD]
    spread = L.max(axis=0) - L.min(axis=0)
    return float(spread.max()), float(np.sqrt(np.mean(spread ** 2)))


def match(levels: list, current_auto: list, snr: list | None = None, hold_below: float = 0.0,
          bass_idx: list | None = None, no_boost: list | None = None) -> list:
    """Corrección por parlante para que todos tengan la misma curva en el punto de escucha.
    Las medidas ya incluyen la corrección actual, así que se corrige solo lo que falta.
    Las bandas que algún parlante no supera sobre el ruido del cuarto se dejan como están."""
    L = np.array([np.asarray(b, dtype=float) - loudness(b) for b in levels])   # sin volumen general
    cur = np.array(current_auto, dtype=float)
    reliable = np.ones(L.shape[1], bool) if snr is None else (np.array(snr) >= MIN_SNR).all(axis=0)
    target = np.empty(L.shape[1])
    for b in range(L.shape[1]):
        # Lo que cada parlante "da" sin su corrección actual, y el rango que puede alcanzar con ella.
        raw = L[:, b] - cur[:, b]
        lo, hi = (raw + MAX_CUT).max(), (raw + MAX_BOOST).min()
        mid = float(np.median(L[:, b]))
        # Objetivo alcanzable por todos, lo más cerca posible de la mediana; prefiere recortar a empujar.
        target[b] = min(max(mid, lo), hi) if lo <= hi else (lo + hi) / 2
    step = 0.85 * (target[None, :] - L)          # amortiguado: la medida real trae algo de ruido
    step[:, ~reliable] = 0.0
    low = _f < hold_below            # bajo el cruce los graves los da otro parlante: no se igualan
    # Suavizado leve del paso entre bandas vecinas: evita "dientes" por ruido de medición.
    sm = step.copy()
    sm[:, 1:-1] = 0.25 * step[:, :-2] + 0.5 * step[:, 1:-1] + 0.25 * step[:, 2:]
    new = np.clip(cur + sm, MAX_CUT, MAX_BOOST)
    new[:, low] = 0.0
    for i, mask in enumerate(no_boost or []):        # parlantes pequeños: en su zona débil solo bajar
        new[i, mask] = np.minimum(new[i, mask], 0.0)
    # El micrófono de la MacBook capta poco bajo ~140 Hz: ahí un parlante «parece» flojo aunque no lo
    # sea. Subir por eso lo satura, así que en esa zona la igualación solo puede bajar.
    new[:, _f < MIC_LOW_HZ] = np.minimum(new[:, _f < MIC_LOW_HZ], 0.0)
    # Bajo el cruce, quien lleva los graves se aplana hacia su propio nivel de medios (sin retumbe);
    # el resto no se toca porque ahí ya no suena.
    for i in (bass_idx or []):
        raw = np.asarray(levels[i], dtype=float) - cur[i]    # respuesta sin corrección
        rel = raw[low] - raw[(_f >= 250) & (_f <= 1000)].mean()   # relativa a sus propios medios
        rel = np.where(np.array(snr)[i, low] >= MIN_SNR, rel, 0.0) if snr is not None else rel
        # Solo la mitad y con tope de 6 dB: el micrófono de la Mac exagera los graves de un parlante cercano
        # (midió +17 dB en el Bose) y aplanarlo entero dejaba huecos de −14 dB en 63–100 Hz con 50 Hz intacto.
        c = np.clip(-0.5 * rel, -BASS_FLAT_MAX, BASS_FLAT_MAX / 2)
        c[1:-1] = 0.25 * c[:-2] + 0.5 * c[1:-1] + 0.25 * c[2:]
        new[i, low] = c
    return [[round(float(x), 2) for x in row] for row in new]


def neutral(curves: list, no_boost: list | None = None) -> list:
    """Deja en cada curva solo la FORMA, sin volumen general: el volumen lo iguala la ganancia de la
    calibración. Si la igualación también subía o bajaba todo, las dos se pisaban de pasada en pasada
    (p. ej. un parlante con −9 dB parejos de ecualizador y su ganancia al tope de +6 dB).
    De paso suaviza entre bandas vecinas para que no se acumulen dientes por ruido de medición."""
    C = np.array(curves, dtype=float)
    for i in range(len(C)):
        C[i] -= loudness(C[i])
    C[:, 1:-1] = 0.25 * C[:, :-2] + 0.5 * C[:, 1:-1] + 0.25 * C[:, 2:]
    for i, mask in enumerate(no_boost or []):
        C[i, mask] = np.minimum(C[i, mask], 0.0)
    C[:, _f < MIC_LOW_HZ] = np.minimum(C[:, _f < MIC_LOW_HZ], 0.0)
    C = np.clip(C, MAX_CUT, MAX_BOOST)
    return [[round(float(x), 2) for x in row] for row in C]


def bass_extension(bands, auto) -> float:
    """dB de graves (50–100 Hz) respecto a medios (250 Hz–1 kHz), sin la corrección aplicada."""
    raw = np.asarray(bands, dtype=float) - np.asarray(auto, dtype=float)
    lo = raw[(_f >= 50) & (_f <= 100)]
    mid = raw[(_f >= 250) & (_f <= 1000)]
    return float(10 * np.log10(np.mean(10 ** (lo / 10))) - 10 * np.log10(np.mean(10 ** (mid / 10))))


def low_limit(bands, auto) -> float:
    """Frecuencia más baja (Hz) desde la que el parlante se mantiene a ≤6 dB de sus medios: debajo
    de eso ya no da graves con fuerza (equivale al «f3/f6» de una ficha técnica)."""
    raw = np.asarray(bands, dtype=float) - np.asarray(auto, dtype=float)
    raw = np.convolve(np.pad(raw, 1, mode="edge"), np.ones(3) / 3, mode="valid")   # pozos de sala fuera
    mid = 10 * np.log10(np.mean(10 ** (raw[(_f >= 250) & (_f <= 1000)] / 10)))
    f = 250.0
    for i in range(int(np.where(_f == 250)[0][0]), -1, -1):
        if raw[i] < mid - 6:
            break
        f = float(_f[i])
    return f


SMALL_BASS_DB = -10.0   # con menos graves que esto respecto a sus medios, es un parlante «pequeño»


def is_small(spk: dict) -> bool:
    return spk.get("bass_db") is not None and spk["bass_db"] < SMALL_BASS_DB


def protect_hz(spk: dict) -> float | None:
    """Frecuencia del filtro de protección: por debajo, el parlante solo distorsiona."""
    if not is_small(spk) or not spk.get("low_hz"):
        return None
    return float(max(90.0, 0.75 * spk["low_hz"]))


def no_boost_mask(spk: dict) -> np.ndarray:
    """Bandas donde la igualación no puede subir (zona débil de un parlante pequeño)."""
    if not is_small(spk) or not spk.get("low_hz"):
        return np.zeros(len(THIRD), bool)
    return _f < 1.6 * spk["low_hz"]


def find_pairs(speakers_cfg: list, listener: dict) -> list:
    """Parlantes que forman pareja: simétricos respecto a tu eje (ángulos ±, misma distancia aprox.)."""
    import math
    az = [math.degrees(math.atan2(sp["x"] - listener["x"], sp["y"] - listener["y"])) for sp in speakers_cfg]
    dist = [math.hypot(sp["x"] - listener["x"], sp["y"] - listener["y"]) for sp in speakers_cfg]
    pairs, used = [], set()
    for i in range(len(az)):
        for j in range(i + 1, len(az)):
            if i in used or j in used or abs(az[i]) < 15 or abs(az[j]) < 15:
                continue
            if az[i] * az[j] < 0 and abs(abs(az[i]) - abs(az[j])) < 25 and \
                    abs(dist[i] - dist[j]) < 0.35 * max(dist[i], dist[j]):
                pairs.append((i, j))
                used |= {i, j}
    return pairs


def pair_floor(a: dict, b: dict) -> float:
    """Desde qué frecuencia igualar una pareja: donde los dos reproducen bien."""
    lows = [x.get("low_hz") for x in (a, b) if x.get("low_hz")]
    return max(lows) * 1.6 if lows else 200.0


def match_pairs(curves: list, auto: list, speakers_cfg: list, pairs: list, snr: list | None = None) -> list:
    """Ajusta cada pareja a su punto medio (banda por banda) para que suenen idénticos entre sí.
    `curves`: respuesta medida (dB) ya con la corrección `auto` aplicada. Prefiere recortar a subir."""
    new = [list(map(float, a)) for a in auto]
    for i, j in pairs:
        ci = np.asarray(curves[i], float) - loudness(curves[i])
        cj = np.asarray(curves[j], float) - loudness(curves[j])
        ok = _f >= pair_floor(speakers_cfg[i], speakers_cfg[j])
        if snr is not None:
            ok &= (np.asarray(snr[i]) >= MIN_SNR) & (np.asarray(snr[j]) >= MIN_SNR)
        d = np.where(ok, ci - cj, 0.0)           # cuánto más suena i que j en cada banda
        # Amortiguado y suavizado entre bandas vecinas, como en match(): una sola medición trae ±1–2 dB de
        # ruido por banda y corregirlo entero, pasada tras pasada, dejaba curvas con dientes de ±11 dB.
        d = 0.85 * d
        d[1:-1] = 0.25 * d[:-2] + 0.5 * d[1:-1] + 0.25 * d[2:]
        ai = np.asarray(new[i]) - d / 2
        aj = np.asarray(new[j]) + d / 2
        over = np.maximum(np.maximum(ai, aj) - MAX_BOOST, 0)   # si alguno pasaría el tope, bajar los dos
        ai, aj = np.clip(ai - over, MAX_CUT, MAX_BOOST), np.clip(aj - over, MAX_CUT, MAX_BOOST)
        new[i] = [round(float(x), 2) for x in ai]
        new[j] = [round(float(x), 2) for x in aj]
    return new
