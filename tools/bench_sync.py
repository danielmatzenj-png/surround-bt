"""Banco de pruebas de sincronía (en silencio, por BlackHole). Uso: python3 tools/bench_sync.py [segundos]

Mide:
1. Estabilidad del retraso: cada salida emite un tono piloto propio (125 / 160 Hz) generado con la hora
   de pared; en la entrada de BlackHole se mide su fase cada segundo → retraso en el tiempo.
   Desviación y rango (ms) = cuánto «baila» la salida.
2. Desfase entre las dos salidas: diferencia de los dos retrasos (lo que se oye como desincronía).
3. Calidad del remuestreo (sin audio): un tono de 15 kHz leído con ratio ≠ 1 → nivel y ruido (dB).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbt.engine import Engine                     # noqa: E402
from sbt.fifo import Fifo                         # noqa: E402
from sbt.monitor import HumGen, HumTone, RoomMonitor   # noqa: E402
from sbt.sources import SR, SilenceSource         # noqa: E402

PILOTS = [125.0, 160.0]


def resampler_quality():
    """Tono de 15 kHz leído con ratio ≠ 1 (lo que hace el control de deriva): relación señal/ruido (dB)."""
    out = {}
    for ratio in (1.0003, 0.9991):
        f = Fifo(SR * 4)
        t = np.arange(SR * 3) / SR
        f.write((0.5 * np.sin(2 * np.pi * 15000 * t)).astype(np.float32))
        y = np.concatenate([f.read(512, ratio) for _ in range(150)])[4096:]
        w = np.blackman(len(y))
        Y = np.abs(np.fft.rfft(y * w)) ** 2
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        k = int(np.argmin(abs(fr - 15000 * ratio)))      # al leer más rápido, el tono sube
        sig = Y[max(0, k - 8):k + 9].sum()
        out[ratio] = round(float(10 * np.log10(sig / (Y.sum() - sig + 1e-20))), 1)
    return out


def run(seconds=60):
    cfg = {"listener": {"x": 2, "y": 2.4}, "sound": {"bass_mode": "off", "loudness": False, "autosync": False},
           "speakers": [{"name": "A", "device": "BlackHole", "x": 1, "y": 4},
                        {"name": "B", "device": "BlackHole", "x": 3, "y": 4}]}
    eng = Engine(cfg, SilenceSource())
    eng.start()
    eng.wait_settled()
    mon = RoomMonitor(eng, cfg, mic="BlackHole")
    mon.running = False                     # solo usamos su captura del micrófono
    gens = []
    for i, f in enumerate(PILOTS):
        g = HumGen(i, mon.T0)
        tn = HumTone(f)
        tn.w = 0.05 + 0j
        g.tones = [tn]
        gens.append(g)

    class Multi:
        idx = 0
        def add(self, mix, g0, t0):
            for g in gens:
                g.add(mix, g0, t0)
    eng.hum = Multi()
    time.sleep(5)
    series = {f: [] for f in PILOTS}
    fills = {s.name: [] for s in eng.speakers}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        time.sleep(1.0)
        now = time.monotonic()
        for f in PILOTS:
            z = mon._demod(f, now - 1.3, now - 0.3)
            if z is not None:
                series[f].append(np.angle(z))
        for st in eng.status():
            fills[st["name"]].append(st["fill_err_ms"])
    mon.stream.stop()
    eng.stop()
    delays = {}
    for f in PILOTS:
        ph = np.unwrap(np.array(series[f]))
        d = -ph / (2 * np.pi * f) * 1000          # ms (más fase negativa = más retraso)
        delays[f] = d - d.mean()
    a, b = delays[PILOTS[0]], delays[PILOTS[1]]
    n = min(len(a), len(b))
    rel = a[:n] - b[:n]
    res = {
        "salida A: retraso, desviación ms": round(float(np.std(a)), 2),
        "salida A: rango ms": round(float(np.ptp(a)), 2),
        "salida B: retraso, desviación ms": round(float(np.std(b)), 2),
        "A−B (desincronía): desviación ms": round(float(np.std(rel)), 2),
        "A−B (desincronía): rango ms": round(float(np.ptp(rel)), 2),
        "buffer: desviación ms": {k: round(float(np.std(v)), 2) for k, v in fills.items()},
    }
    return res


if __name__ == "__main__":
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print("remuestreo (15 kHz):", resampler_quality())
    for k, v in run(secs).items():
        print(f"{k}: {v}")
