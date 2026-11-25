"""Procesos del motor: niveles por canal, voces claras, modo noche, gestión de graves y dynamic EQ."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .sources_base import SR

DEFAULT_SOUND = {
    "center_db": 0.0, "surround_db": 0.0, "lfe_db": 0.0,   # niveles por canal (como en un receptor AV)
    "dialogue": 0,          # 0 apagado, 1 medio, 2 alto
    "night": False,         # compresor: explosiones más bajas, diálogos igual de claros
    "loudness": True,       # dynamic EQ: más graves/agudos a volumen bajo (el oído los pierde)
    "bass_mode": "auto",    # "auto": graves al parlante que mejor los da; "off": cada uno lo suyo
    "crossover": 100,       # Hz
    "crossover_auto": True, # el cruce lo decide la calibración según los parlantes
    "adaptive": False,      # volumen adaptativo al ruido del cuarto (monitor con el micrófono)
    "mask": False,          # enmascarar el ruido del cuarto con un ruido suave de su misma forma
    "mask_db": 0.0,         # intensidad de la máscara respecto al ruido medido
    "hum": False,           # cancelador de zumbidos (experimental)
    "spatial": 1,           # espacialidad: 0 natural, 1 amplio, 2 envolvente, 3 máximo
    "autosync": True,       # corrige sola la sincronía mientras suena (escucha con el micrófono)
    "party": False,         # modo fiesta: todos los parlantes suenan con todo (sin surround)
    "pair_volume": True,    # parejas (p. ej. los laterales) siempre con el mismo ajuste de volumen
}


def auto_crossover(speakers_cfg: list, bass_idx: list) -> int | None:
    """Cruce = donde el lateral más limitado deja de dar graves (redondeado, 80–200 Hz)."""
    lims = [s.get("low_hz") for i, s in enumerate(speakers_cfg) if i not in bass_idx]
    lims = [x for x in lims if x]
    if not lims or not bass_idx:
        return None
    return int(min(200, max(80, round(max(lims) * 1.1 / 10) * 10)))


def sound_cfg(cfg: dict) -> dict:
    return {**DEFAULT_SOUND, **cfg.get("sound", {})}


class Filter:
    """Filtro IIR con estado, para N columnas."""

    def __init__(self, sos: np.ndarray, cols: int):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2, cols))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def lr4(kind: str, fc: float, cols: int) -> Filter:
    """Linkwitz-Riley de 4.º orden: pasabajos + pasaaltos suman plano en el cruce."""
    b = butter(2, fc, kind, fs=SR, output="sos")
    return Filter(np.vstack([b, b]), cols)


def peaking(f0: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * a, -2 * np.cos(w), 1 - al * a])
    d = np.array([1 + al / a, -2 * np.cos(w), 1 - al / a])
    return np.concatenate([b / d[0], d / d[0]])[None, :]


class Compressor:
    """Compresor enlazado (todas las columnas reciben la misma ganancia), por sub-bloques de 64."""

    def __init__(self, thresh_db=-30.0, ratio=4.0, makeup_db=7.0, attack=0.005, release=0.25):
        self.t, self.r, self.mk = thresh_db, ratio, 10 ** (makeup_db / 20)
        self.att = np.exp(-64 / (attack * SR))
        self.rel = np.exp(-64 / (release * SR))
        self.env = 1e-6
        self.g = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for i in range(0, len(x), 64):
            seg = x[i:i + 64]
            lvl = float(np.sqrt(np.mean(seg ** 2)) * np.sqrt(2)) + 1e-9
            c = self.att if lvl > self.env else self.rel
            self.env = c * self.env + (1 - c) * lvl
            db = 20 * np.log10(self.env)
            over = max(0.0, db - self.t)
            g_new = 10 ** (-(over - over / self.r) / 20) * self.mk
            ramp = np.linspace(self.g, g_new, len(seg), endpoint=False, dtype=np.float32)[:, None]
            out[i:i + 64] = seg * ramp
            self.g = g_new
        return out


def loudness_bands(volume_db: float) -> list:
    """Dynamic EQ: refuerzo para las 8 bandas del usuario según el volumen (curvas de igual sonoridad).
    A 0 dB no hace nada; a −30 dB suma ~+7 dB en graves y ~+2 dB en agudos."""
    down = max(0.0, -float(volume_db))
    bass = min(8.0, down * 0.24)
    treble = min(3.0, down * 0.07)
    return [bass, bass * 0.6, bass * 0.2, 0, 0, 0, treble * 0.5, treble]


def bass_speakers(speakers_cfg: list) -> list:
    """Índices de los parlantes que reproducen bien los graves (medido al calibrar: `bass_db`).
    Se eligen los que están a ≤6 dB del mejor. Vacío si no hay medición."""
    vals = [s.get("bass_db") for s in speakers_cfg]
    if not vals or any(v is None for v in vals) or len(vals) < 2:
        return []
    best = max(vals)
    return [i for i, v in enumerate(vals) if v >= best - 6.0]


class SoundProcessor:
    """Cadena antes y después del reparto a parlantes (ver Engine._push)."""

    def __init__(self, sound: dict, n_speakers: int, bass_idx: list):
        self.set(sound, n_speakers, bass_idx)

    def set(self, sound: dict, n_speakers: int, bass_idx: list):
        self.s = {**DEFAULT_SOUND, **sound}
        s = self.s
        dlg = int(s["dialogue"])
        self.ch_gain = np.ones(6, np.float32)
        self.ch_gain[2] = 10 ** ((s["center_db"] + 3.0 * dlg) / 20)
        self.ch_gain[4] = self.ch_gain[5] = 10 ** ((s["surround_db"] - 1.5 * dlg) / 20)
        # LFE de películas viene grabado 10 dB abajo; se repone +6 (parlantes pequeños) + ajuste.
        self.lfe_gain = 10 ** ((s["lfe_db"] + 6.0) / 20)
        self.presence = Filter(peaking(2500, 2.5 * dlg, 0.9), 1) if dlg else None
        self.comp = Compressor() if s["night"] else None
        self.bass_idx = list(bass_idx) if s["bass_mode"] == "auto" else []
        fc = float(s["crossover"])
        if self.bass_idx:
            self.hp = lr4("highpass", fc, n_speakers)
            self.lp = lr4("lowpass", fc, 1)
        else:
            self.hp = self.lp = None

    @property
    def managing_bass(self) -> bool:
        return bool(self.bass_idx)

    def channels(self, blk: np.ndarray) -> np.ndarray:
        """Sobre los 6 canales: niveles, voces claras y modo noche."""
        x = blk * self.ch_gain
        if self.presence is not None:
            x[:, 2:3] = self.presence(x[:, 2:3])
        if self.comp is not None:
            x = self.comp(x)
        return x

    def speakers(self, feeds: np.ndarray, lfe: np.ndarray) -> np.ndarray:
        """Sobre las señales de cada parlante: los graves de todos (y el LFE) van a los parlantes
        que mejor los reproducen; los demás quedan sin graves y suenan más limpios."""
        if not self.bass_idx:
            return feeds
        bus = self.lp((feeds.sum(axis=1) + lfe * self.lfe_gain)[:, None])[:, 0]
        out = self.hp(feeds)
        out[:, self.bass_idx] += (bus / len(self.bass_idx))[:, None]
        return out


class BassLimiter:
    """Limitador solo para graves (<150 Hz) de un parlante: si se acercan a lo que el cono aguanta, los
    baja unos milisegundos. El resto del sonido no se toca."""

    def __init__(self, fc: float = 150.0, thresh: float = 0.35, attack: float = 0.002, release: float = 0.12):
        self.lp = lr4("lowpass", fc, 1)
        self.hp = lr4("highpass", fc, 1)          # LR4: separa bien y la suma es plana
        self.t = thresh
        self.att = np.exp(-1 / (attack * SR / 32))
        self.rel = np.exp(-1 / (release * SR / 32))
        self.g = 1.0
        self.active = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        low = self.lp(x[:, None])[:, 0]
        rest = self.hp(x[:, None])[:, 0]
        out = np.empty_like(low)
        hits = 0
        for i in range(0, len(low), 32):
            seg = low[i:i + 32]
            pk = float(np.abs(seg).max(initial=0.0)) * self.g
            want = min(1.0, self.t / pk * self.g) if pk > self.t else 1.0
            c = self.att if want < self.g else self.rel
            g_new = c * self.g + (1 - c) * want
            out[i:i + 32] = seg * np.linspace(self.g, g_new, len(seg), endpoint=False)
            hits += g_new < 0.97
            self.g = g_new
        self.active = hits / max(1, (len(low) + 31) // 32)
        return rest + out


# Espacialidad: ángulo de L/R, ambiente que va a los costados/atrás y el que queda adelante.
SPATIAL = [
    {"width": 30, "surround": 0.85, "front_amb": 0.50},   # natural (estándar ITU)
    {"width": 40, "surround": 1.00, "front_amb": 0.40},   # amplio
    {"width": 50, "surround": 1.20, "front_amb": 0.30},   # envolvente
    {"width": 60, "surround": 1.40, "front_amb": 0.20},   # máximo
]


def spatial(sound: dict) -> dict:
    return SPATIAL[max(0, min(3, int(sound.get("spatial", 1))))]


SCENES = {
    "cine":      {"sound": {"spatial": 2, "dialogue": 1, "night": False, "loudness": True}, "eq": "cinema"},
    "concierto": {"sound": {"spatial": 3, "dialogue": 0, "night": False, "loudness": True}, "eq": "music"},
    "juego":     {"sound": {"spatial": 3, "dialogue": 0, "night": False, "loudness": True}, "eq": "flat"},
    "noticias":  {"sound": {"spatial": 0, "dialogue": 2, "night": False, "loudness": True}, "eq": "dialogue"},
    "noche":     {"sound": {"spatial": 1, "dialogue": 1, "night": True, "loudness": True}, "eq": "night"},
}


# El piso cambia los agudos que llegan: la alfombra los absorbe, la baldosa los refleja (8 bandas del usuario).
FLOOR_TILT = {"madera": [0] * 8, "alfombra": [0, 0, 0, 0, 0, 0.5, 1.0, 1.5], "baldosa": [0, 0, 0, 0, 0, -0.5, -1.0, -1.0]}
