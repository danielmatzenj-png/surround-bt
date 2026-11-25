"""Calibración con micrófono: cada parlante emite un barrido, se mide cuándo y qué tan fuerte llega
al punto de escucha, y se igualan retrasos y volúmenes."""
from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

from . import dsp
from . import eq as eqmod
from .clock import adc_time
from .sources import SR

SPACING = 1.6     # segundos entre barridos
CANCEL = threading.Event()   # el servidor lo activa con «Cancelar»


class Cancelled(Exception):
    pass
WINDOW = 1.0      # ventana de búsqueda tras el instante esperado (latencia BT máx. razonable)


def make_sweep(dur=0.6, f0=45.0, f1=12000.0, amp=0.15) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    k = np.log(f1 / f0)
    sig = np.sin(2 * np.pi * f0 * dur / k * (np.exp(t / dur * k) - 1))
    fade = int(0.01 * SR)
    env = np.ones_like(sig)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (amp * sig * env).astype(np.float32)


def find_input_device(pattern=None) -> int:
    if pattern is None:
        # Nunca el micrófono de un parlante Bluetooth (lo pasaría a modo llamada): el de la Mac o uno USB.
        from . import coreaudio
        want = coreaudio.preferred_mic()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and d["name"] == want:
                return i
        return sd.default.device[0]
    pat = str(pattern).lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and pat in d["name"].lower():
            return i
    raise RuntimeError(f"No encontré micrófono que contenga '{pattern}'")


def measure(engine, mic=None, rounds=2, log=print) -> list:
    """Devuelve por parlante: {'latency_ms', 'level_db', 'snr'} (latencias relativas entre sí)."""
    if CANCEL.is_set():
        raise Cancelled()
    sweep = make_sweep()
    n = len(engine.speakers)
    chunks = []

    start = []                                  # hora exacta de la primera muestra (CoreAudio)

    def cb(indata, frames, t, status):
        if not chunks:
            start.append(adc_time(t))
        chunks.append(indata[:, 0].copy())

    mic_dev = find_input_device(mic)
    mic_sr = float(sd.query_devices(mic_dev)["default_samplerate"])
    stream = sd.InputStream(device=mic_dev, channels=1, samplerate=mic_sr,
                            dtype="float32", callback=cb)
    stream.start()
    mic_t0 = time.monotonic()
    time.sleep(1.0)   # el micrófono recién abierto tarda en estabilizarse; antes falsea los graves

    f0 = engine.produced + int(0.4 * SR)
    plan = []
    for r in range(rounds):
        for i in range(n):
            at = f0 + int(((r * n) + i) * SPACING * SR)
            engine.inject(i, sweep, at)
            plan.append((i, at))
    total = (len(plan) * SPACING) + 0.4 + WINDOW + 0.5
    log(f"Midiendo {n} parlante(s) x {rounds} rondas (~{total:.0f} s). Silencio en el cuarto, por favor…")
    time.sleep(total)
    stream.stop()
    stream.close()

    rec = np.concatenate(chunks)
    # La hora de abrir el micrófono no es la de su primera muestra (±1–2 ms según cada apertura); la
    # que informa CoreAudio sí. Sin ella (algún dispositivo raro), queda la aproximada.
    if start and start[0] is not None:
        mic_t0 = start[0]
    if mic_sr != SR:
        from fractions import Fraction
        fr = Fraction(SR / mic_sr).limit_denominator(1000)
        rec = resample_poly(rec, fr.numerator, fr.denominator).astype(np.float32)
    # La llegada se detecta solo con >1 kHz: los graves (resonancias del parlante y del cuarto)
    # deforman la correlación y la corren varios milisegundos.
    hp = butter(4, 1000, "highpass", fs=SR, output="sos")
    corr = np.abs(fftconvolve(sosfiltfilt(hp, rec), sosfiltfilt(hp, sweep)[::-1], mode="full"))
    L = len(sweep) - 1
    # Ruido del cuarto: el tramo en silencio antes del primer barrido, del mismo largo que cada toma.
    take = len(sweep) + int(0.15 * SR)
    noise_bands = np.array(eqmod.band_levels(rec[int(0.45 * SR):int(0.45 * SR) + take], sweep))

    per = {i: [] for i in range(n)}
    for i, at in plan:
        expected = int((engine.frame_to_wall(at) - mic_t0) * SR) + L
        a = max(expected, 0)
        b = min(expected + int(WINDOW * SR), len(corr))
        if b - a < SR // 10:
            continue
        seg = corr[a:b]
        k = int(np.argmax(seg))
        # Primer pico fuerte (sonido directo) en vez del máximo absoluto (puede ser un rebote)
        thresh = 0.5 * seg[k]
        first = int(np.argmax(seg[:k + 1] >= thresh))
        noise = float(np.median(seg)) + 1e-12
        s0 = a + first - L                       # inicio del barrido en la grabación
        grab = rec[max(s0 - int(0.05 * SR), 0):s0 + len(sweep) + int(0.1 * SR)]  # margen: los graves van primero
        bands = eqmod.band_levels(grab, sweep)
        per[i].append({"lat": (a + first - expected) / SR,
                       "bands": bands,
                       "band_snr": np.array(bands) - noise_bands,
                       "snr": float(seg[k] / noise)})

    results = []
    for i in range(n):
        m = per[i]
        if not m:
            results.append(None)
            continue
        bands = np.median([x["bands"] for x in m], axis=0)
        results.append({
            "latency_ms": float(np.median([x["lat"] for x in m]) * 1000),
            "level_db": eqmod.loudness(bands),             # volumen en 125 Hz–8 kHz
            "bands": [float(b) for b in bands],
            "band_snr": [float(x) for x in np.median([x["band_snr"] for x in m], axis=0)],
            "snr": float(np.median([x["snr"] for x in m])),
        })
    return results


def solve(engine, results, min_snr=12.0):
    """Calcula nuevos retrasos/ganancias en ms/dB. Las medidas ya incluyen el retraso actual de cada parlante."""
    ok = [i for i, r in enumerate(results) if r and r["snr"] >= min_snr]
    if not ok:
        raise RuntimeError("Ningún parlante se escuchó claro en el micrófono. Sube el volumen o acerca la Mac.")
    cur_delay = np.array([s.delay / SR * 1000 for s in engine.speakers])
    cur_gain = np.array([20 * np.log10(s.gain) for s in engine.speakers])
    total = np.array([results[i]["latency_ms"] if i in ok else np.nan for i in range(len(results))])
    levels = np.array([results[i]["level_db"] if i in ok else np.nan for i in range(len(results))])
    worst = np.nanmax(total)
    new_delay = np.where(np.isnan(total), cur_delay, cur_delay + (worst - total))
    new_delay -= new_delay.min()
    quiet = np.nanmin(levels)
    new_gain = np.where(np.isnan(levels), cur_gain, cur_gain + (quiet - levels))
    new_gain -= new_gain.max()
    new_gain = np.clip(new_gain, -30, 0)
    return new_delay, new_gain, ok


TOLERANCE = {"level_db": 0.3, "tone_db": 0.7, "time_ms": 1.0}


def _metrics(res: list) -> dict:
    lv = [r["level_db"] for r in res]
    lat = [r["latency_ms"] for r in res]
    tmax, trms = eqmod.tone_spread([r["bands"] for r in res]) if len(res) > 1 else (0.0, 0.0)
    return {"level_db": max(lv) - min(lv), "time_ms": max(lat) - min(lat), "tone_db": trms, "tone_max_db": tmax}


def _report(eng, res: list, m: dict) -> dict:
    mean = float(np.mean([r["level_db"] for r in res]))
    return {"bands": eqmod.THIRD, "time_ms": round(m["time_ms"], 1), "level_db": round(m["level_db"], 1),
            "tone_db": round(m["tone_db"], 1), "tone_max_db": round(m["tone_max_db"], 1),
            "curves": {s.name: [round(float(x) - mean, 1) for x in r["bands"]] for s, r in zip(eng.speakers, res)},
            "snr_ok": [bool(x) for x in (np.array([r["band_snr"] for r in res]) >= eqmod.MIN_SNR).all(axis=0)],
            "at": time.time()}


SPEED = 343.0


def geometry_offsets(cfg: dict) -> list:
    """Por parlante: (ms, dB) para pasar de lo medido en el micrófono a lo que llega a la cabeza.
    El nivel usa campo directo + reverberante (distancia crítica estimada del tamaño del cuarto)."""
    mic = cfg.get("mic_pos")
    lst = cfg["listener"]
    if not mic:
        return [(0.0, 0.0)] * len(cfg["speakers"])
    room = cfg.get("room", {"width": 4, "depth": 5})
    vol = room["width"] * room["depth"] * 2.5
    rc = 0.057 * np.sqrt(vol / 0.5)          # distancia crítica, RT60 ~0,5 s de una sala amoblada
    out = []
    for spk in cfg["speakers"]:
        dm = max(0.2, float(np.hypot(spk["x"] - mic["x"], spk["y"] - mic["y"])))
        dl = max(0.2, float(np.hypot(spk["x"] - lst["x"], spk["y"] - lst["y"])))
        ms = (dl - dm) / SPEED * 1000
        db = 10 * np.log10((1 / dl ** 2 + 1 / rc ** 2) / (1 / dm ** 2 + 1 / rc ** 2))
        out.append((float(ms), float(db)))
    return out


def _adjust(res: list, offsets: list, geo: list) -> list:
    """Aplica la corrección por giro de la Mac (offsets) y por geometría (geo) a una medición."""
    out = []
    for r, off, (gms, gdb) in zip(res, offsets, geo):
        if not r:
            out.append(r)
            continue
        r = dict(r)
        r["latency_ms"] = r["latency_ms"] - off["lat"] + gms
        r["bands"] = [b - ob + gdb for b, ob in zip(r["bands"], off["bands"])]
        r["level_db"] = eqmod.loudness(r["bands"])
        out.append(r)
    return out


def _rotation_offsets(a: list, b: list) -> list:
    """La mitad de la diferencia entre las dos orientaciones: lo que el micrófono 'desvía' estando girado."""
    out = []
    for ra, rb in zip(a, b):
        if not ra or not rb:
            out.append({"lat": 0.0, "bands": [0.0] * len(eqmod.THIRD)})
            continue
        lat = (rb["latency_ms"] - ra["latency_ms"]) / 2
        bands = np.array([(y - x) / 2 for x, y in zip(ra["bands"], rb["bands"])])
        # Físicamente, el micrófono dentro del chasis no desvía más de ~20 cm (≈0,6 ms) ni la sombra
        # de la pantalla más de unos dB. Más que eso es una medición inestable: no se usa.
        if abs(lat) > 0.8:
            lat = 0.0
        bands = np.where(np.abs(bands) > 4.0, 0.0, bands)
        out.append({"lat": float(lat), "bands": [float(b) for b in bands], "rejected": lat == 0.0})
    return out


def wait_stable(eng, cfg, log=print, tol_ms=2.0, max_s=75.0):
    """El retraso Bluetooth se asienta tras empezar a sonar (medido: hasta ~40 s y 20 ms). Mide
    tiempos hasta que dos mediciones seguidas coinciden por parlante."""
    t_end = time.monotonic() + max_s
    prev = None
    log("Esperando que el Bluetooth se estabilice…")
    while time.monotonic() < t_end:
        res = measure(eng, cfg.get("mic"), rounds=1, log=lambda *a: None)
        cur = [r["latency_ms"] if r else None for r in res]
        if prev and all(a is not None and b is not None for a, b in zip(prev, cur)):
            delta = max(abs(a - b) for a, b in zip(prev, cur))
            if delta <= tol_ms:
                log(f"  Estable (variación {delta:.1f} ms).")
                return
            log(f"  Todavía se mueve ({delta:.1f} ms)…")
        prev = cur
    log("  Sigue moviéndose un poco; se calibra igual.")


def run(cfg: dict, max_passes=6, rounds=3, log=print, ask=None, engine=None) -> dict:
    """Calibración adaptativa: mide, corrige y repite hasta quedar dentro de TOLERANCE (o max_passes).
    La medición que cumple la tolerancia es a la vez la verificación final.
    Si se pasa `ask`, tras la primera medición pide girar la Mac 180° en el mismo lugar: promediar
    ambas orientaciones cancela dónde está el micrófono dentro del chasis (y parte de la sombra de la
    pantalla). Desde ahí cada medición se corrige con esa diferencia."""
    import math
    from .engine import Engine
    from .sources import SilenceSource
    own = engine is None
    eng = Engine(cfg, SilenceSource()) if own else engine
    for s_ in eng.speakers:
        s_.gain = 10 ** (s_.cal_gain_db / 20)
    if own:
        eng.start()
    report = None
    try:
        time.sleep(1.0)
        wait_stable(eng, cfg, log)
        geo = geometry_offsets(cfg)
        if cfg.get("mic_pos"):
            log("Corrigiendo por la posición de la Mac: " + ", ".join(
                f"{sp['name']} {ms:+.1f} ms / {db:+.1f} dB" for sp, (ms, db) in zip(cfg["speakers"], geo)))
        offsets = [{"lat": 0.0, "bands": [0.0] * len(eqmod.THIRD)}] * len(eng.speakers)
        for pasada in range(max_passes + 1):
            res = measure(eng, cfg.get("mic"), rounds=rounds, log=log)
            if pasada == 0 and ask:
                ask("Gira la Mac media vuelta (180°) en el mismo lugar y toca «Ya la giré».")
                log("Midiendo con la Mac girada…")
                res_b = measure(eng, cfg.get("mic"), rounds=rounds, log=log)
                offsets = _rotation_offsets(res, res_b)
                for sp, off in zip(eng.speakers, offsets):
                    note = " (tiempo inestable, no se usa)" if off.get("rejected") else ""
                    log(f"  Efecto de la posición del micrófono en {sp.name}: {off['lat']:+.2f} ms, "
                        f"{np.mean(off['bands']):+.2f} dB{note}")
                res = res_b
            res = _adjust(res, offsets, geo)
            for s, r in zip(eng.speakers, res):
                if r:
                    log(f"  {s.name}: llega {r['latency_ms']:+7.1f} ms, nivel {r['level_db']:6.1f} dB, "
                        f"claridad {r['snr']:.0f}x")
                else:
                    log(f"  {s.name}: no se escuchó")
            delays, gains, ok = solve(eng, res)
            complete = len(ok) == len(res)
            if complete:
                m = _metrics(res)
                log(f"Pasada {pasada + 1}: tiempo ±{m['time_ms']:.1f} ms, volumen ±{m['level_db']:.1f} dB, "
                    f"tono ±{m['tone_db']:.1f} dB (máx {m['tone_max_db']:.1f})")
                report = _report(eng, res, m)
                for s_, r in zip(eng.speakers, res):     # para el monitor del cuarto
                    s_.cfg["tf_db"] = [round(float(x), 2) for x in r["bands"]]
                cfg["acoustic_latency_ms"] = round(float(np.mean([r["latency_ms"] for r in res])), 1)
                for s_, r in zip(eng.speakers, res):   # capacidad de graves (para el gestor de graves)
                    a_ = eqmod.auto_for(s_.cfg, cfg)
                    s_.cfg["bass_db"] = round(eqmod.bass_extension(r["bands"], a_), 1)
                    s_.cfg["low_hz"] = eqmod.low_limit(r["bands"], a_)
                snd = dsp.sound_cfg(cfg)
                if snd["crossover_auto"] and snd["bass_mode"] == "auto":
                    xo = dsp.auto_crossover([s_.cfg for s_ in eng.speakers],
                                            dsp.bass_speakers([s_.cfg for s_ in eng.speakers]))
                    if xo and xo != snd["crossover"]:
                        cfg["sound"] = {**snd, "crossover": xo}
                        log(f"  Cruce de graves: {xo} Hz")
                if all(m[k] <= v for k, v in TOLERANCE.items()):
                    log(f"Dentro de la tolerancia en {pasada + 1} pasada(s).")
                    break
            if pasada == max_passes:
                log("Se alcanzó el máximo de pasadas; queda el mejor resultado medido.")
                break
            if len(ok) > 1 and gains.min() <= -10:
                loud = eng.speakers[int(np.argmin(gains))].name
                log(f"  Aviso: {loud} suena {-gains.min():.0f} dB más fuerte que el resto. Para no perder volumen, "
                    f"baja {loud} con sus botones (o sube los demás) y vuelve a calibrar.")
            for s, d, g in zip(eng.speakers, delays, gains):
                s.set_delay(int(round(d * SR / 1000)))
                s.gain = 10 ** (g / 20)
            if cfg.get("eq_match", True) and complete and len(res) > 1:
                cur = [eqmod.auto_for(s.cfg, cfg) for s in eng.speakers]
                snd = dsp.sound_cfg(cfg)
                hold = snd["crossover"] * 1.25 if snd["bass_mode"] == "auto" and any(
                    sc.get("bass_db") is not None for sc in cfg["speakers"]) else 0.0
                new = eqmod.match([r["bands"] for r in res], cur, [r["band_snr"] for r in res], hold,
                                  dsp.bass_speakers([s_.cfg for s_ in eng.speakers]) if hold else None,
                                  [eqmod.no_boost_mask(s_.cfg) for s_ in eng.speakers])
                # Parejas (p. ej. los laterales): además, iguales entre sí, banda por banda.
                pairs = eqmod.find_pairs([s_.cfg for s_ in eng.speakers], cfg["listener"])
                if pairs:
                    # lo que se oiría con la corrección nueva = medido + (nueva − actual)
                    pred = [np.asarray(r["bands"]) + np.asarray(nw) - np.asarray(c)
                            for r, nw, c in zip(res, new, cur)]
                    new = eqmod.match_pairs(pred, new, [s_.cfg for s_ in eng.speakers], pairs,
                                            [r["band_snr"] for r in res])
                new = eqmod.neutral(new, [eqmod.no_boost_mask(s_.cfg) for s_ in eng.speakers])
                for s, curve in zip(eng.speakers, new):
                    s.cfg["eq_auto"] = curve
                eng.apply_eq(cfg)
            eng.wait_settled()
            time.sleep(0.5)
    finally:
        if own:
            eng.stop()
    if report:
        cfg["calibration_report"] = report
        log(f"Resultado: volumen ±{report['level_db']} dB, tono ±{report['tone_db']} dB "
            f"(máx {report['tone_max_db']}), tiempo ±{report['time_ms']} ms")
    for spk, s in zip(cfg["speakers"], eng.speakers):
        spk["delay_ms"] = round(s.delay / SR * 1000, 2)
        spk["gain_db"] = round(20 * math.log10(s.gain), 2)
        if "eq_auto" in s.cfg:
            spk["eq_auto"] = s.cfg["eq_auto"]
        for k in ("bass_db", "low_hz", "tf_db"):
            if k in s.cfg:
                spk[k] = s.cfg[k]
        log(f"{spk['name']}: retraso {spk['delay_ms']} ms, ganancia {spk['gain_db']} dB")
    return cfg


def quick_sync(cfg: dict, log=print, max_passes=3, engine=None) -> dict:
    """Solo tiempos (≈15 s): para cuando un parlante se reconecta y su retraso Bluetooth cambió.
    No toca volumen ni tono."""
    from .engine import Engine
    from .sources import SilenceSource
    own = engine is None
    eng = Engine(cfg, SilenceSource()) if own else engine
    if own:
        eng.start()
    geo = geometry_offsets(cfg)
    zero = [{"lat": 0.0, "bands": [0.0] * len(eqmod.THIRD)}] * len(eng.speakers)
    lats = []
    try:
        time.sleep(1.0)
        wait_stable(eng, cfg, log)
        for p in range(max_passes):
            res = _adjust(measure(eng, cfg.get("mic"), rounds=1, log=log), zero, geo)
            delays, _, ok = solve(eng, res)
            lats = [res[i]["latency_ms"] for i in ok]
            spread = max(lats) - min(lats)
            log(f"Pasada {p + 1}: tiempo ±{spread:.1f} ms")
            if spread <= TOLERANCE["time_ms"]:
                break
            for s, d in zip(eng.speakers, delays):
                s.set_delay(int(round(d * SR / 1000)))
            eng.wait_settled()
            time.sleep(0.3)
    finally:
        if own:
            eng.stop()
    for spk, s in zip(cfg["speakers"], eng.speakers):
        spk["delay_ms"] = round(s.delay / SR * 1000, 2)
    if lats:
        cfg["acoustic_latency_ms"] = round(float(np.mean(lats)), 1)   # para el video y el monitor
    log("Sincronía actualizada.")
    return cfg


def chime() -> np.ndarray:
    """Dos notas suaves para identificar un parlante."""
    t = np.arange(int(0.22 * SR)) / SR
    env = np.sin(np.pi * t / t[-1]) ** 2
    a = 0.12 * np.sin(2 * np.pi * 880 * t) * env
    b = 0.12 * np.sin(2 * np.pi * 1318.5 * t) * env
    return np.concatenate([a, np.zeros(int(0.05 * SR)), b]).astype(np.float32)
