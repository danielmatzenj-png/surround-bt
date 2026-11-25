"""Estabilidad de salida con clics (en silencio, por BlackHole): el motor emite un clic exacto cada segundo
(en su línea de tiempo); se graban de vuelta y se mide la llegada de cada uno. La variación de
(llegada − momento nominal) es lo que «baila» el retraso de salida."""
import sys, time
from pathlib import Path
import numpy as np
import sounddevice as sd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sbt.engine as E
from sbt.engine import Engine
import os
E.USE_LAT = os.environ.get("LAT", "1") == "1"
E.NATIVE_RATE = os.environ.get("NATIVE", "0") == "1"
E.OUT_BLOCK = int(os.environ.get("BLOCK", "512"))
from sbt.sources import SR, SilenceSource

secs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
names = sys.argv[2:] or ["BlackHole"]
cfg = {"listener": {"x": 2, "y": 2}, "sound": {"bass_mode": "off", "loudness": False},
       "speakers": [{"name": f"S{i}", "device": d, "x": 1 + i, "y": 3} for i, d in enumerate(names)]}
eng = Engine(cfg, SilenceSource())
eng.start(); eng.wait_settled(); time.sleep(3)
info = sd.query_devices("BlackHole 2ch", "input"); msr = float(info["default_samplerate"])
chunks = []
st = sd.InputStream(device="BlackHole 2ch", channels=1, samplerate=msr, callback=lambda i, f, t, s: chunks.append(i[:, 0].copy()))
st.start(); m0 = time.monotonic(); time.sleep(0.5)
click = np.zeros(64, np.float32); click[0] = 0.8
first = eng.produced + SR
for k in range(secs):
    for i in range(len(eng.speakers)):
        eng.inject(i, click, first + k * SR + i * (SR // 2))       # cada salida, medio segundo corrida
time.sleep(secs + 2)
st.stop(); eng.stop()
rec = np.concatenate(chunks)
hits = np.where(rec > 0.5 * np.abs(rec).max())[0]
times = []                                     # hora (s, reloj del micrófono) de cada clic
for h in hits:
    if not times or h / msr - times[-1] > 0.2:
        times.append(h / msr)
times = np.array(times)
for i in range(len(eng.speakers)):
    t_i = times[i::len(eng.speakers)][10:]          # sin los primeros 10 (asentamiento)
    k = np.arange(len(t_i))
    p = np.polyfit(k, t_i, 1)                  # la recta quita el desfase fijo y la diferencia de relojes
    res = (t_i - np.polyval(p, k)) * 1000
    print(f"{names[i]}: {len(t_i)} clics | baile del retraso: desvío {res.std():.2f} ms, rango {np.ptp(res):.2f} ms")
if len(eng.speakers) == 2:
    n = min(len(times[0::2]), len(times[1::2]))
    rel = (times[1::2][:n] - times[0::2][:n] - 0.5)[10:] * 1000   # lo que se oye: B respecto de A
    print(f"B−A (desincronía): media {rel.mean():.2f} ms | desvío {rel.std():.2f} ms, rango {np.ptp(rel):.2f} ms")
