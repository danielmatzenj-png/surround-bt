"""Monitor del cuarto: escucha con el micrófono de la Mac mientras suena el sistema.

Lo que oye el micrófono = lo que reproducimos (que conocemos) + el ruido del cuarto. La calibración
midió cuánto llega de cada parlante al micrófono por banda (`tf_db`) y con cuánto retraso
(`acoustic_latency_ms`), así que se puede predecir y restar lo nuestro y quedarse con el ruido.

Con eso:
- Volumen adaptativo: sube volumen y diálogos cuando el ruido sube sobre el fondo habitual.
- Enmascaramiento: ruido suave con la forma espectral del ruido del cuarto, para taparlo.
- Cancelador de zumbidos (experimental): tonos estables de 40–200 Hz. El Bluetooth tarda ~0,4 s, así
  que la antionda no se calcula del ruido "en vivo", sino que se predice (un tono estable es
  predecible) y se ajusta en ciclos midiendo el residuo en el micrófono.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from . import eq as eqmod
from .calibrate import find_input_device
from .clock import adc_time
from .sources_base import SR

NOISE_BANDS = [f for f in eqmod.THIRD if 100 <= f <= 6300]
_T = np.array(eqmod.THIRD, float)
_SEL = np.array([eqmod.THIRD.index(f) for f in NOISE_BANDS])


def _band_energy(x: np.ndarray, sr: float, centers) -> np.ndarray:
    n = 1 << int(np.ceil(np.log2(len(x))))
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    k = 2 ** (1 / 6)
    return np.array([X[(f >= c / k) & (f < c * k)].sum() for c in centers]) + 1e-20


# --- enmascaramiento --------------------------------------------------------------------
class Masker:
    """Ruido descorrelacionado por parlante, filtrado con la forma del ruido del cuarto."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.h = eqmod.fir([0.0] * len(eqmod.THIRD))
        self.gain = 0.0
        self.tails = None

    def shape(self, curve_db, gain: float):
        self.h = eqmod.fir(np.clip(curve_db, -30, 12))
        self.gain = float(gain)

    def block(self, n: int, cols: int) -> np.ndarray:
        from scipy.signal import oaconvolve
        h = self.h
        w = self.rng.standard_normal((n, cols)).astype(np.float32) * self.gain
        y = oaconvolve(w, h[:, None], axes=0)
        if self.tails is None or self.tails.shape != (len(h) - 1, cols):
            self.tails = np.zeros((len(h) - 1, cols), np.float32)
        y[:len(h) - 1] += self.tails
        self.tails = y[n:].astype(np.float32)
        return y[:n].astype(np.float32)


# --- sincronía automática ---------------------------------------------------------------
class SyncTracker:
    """Mientras suena cualquier cosa, estima cuándo llega cada parlante al micrófono.

    Cada parlante recibe una mezcla distinta (centro, laterales, ambiente), así que con varios cuadros
    se puede resolver a la vez la respuesta de cada uno (estimador H1 multicanal: H = Srr⁻¹·Srm por
    frecuencia). La fase de cada respuesta (PHAT, 300 Hz–6 kHz) da un pico en su retraso relativo.
    Si un parlante se corre, su retraso se ajusta poco a poco (sin saltos ni clics)."""

    NF = 8192            # cuadro (170 ms)
    WIN_S = 6.0          # segundos de audio por estimación
    SEARCH_MS = 80.0     # rango de búsqueda alrededor de lo esperado

    def __init__(self):
        self.hist = deque(maxlen=3)
        self.block_until = 0.0
        self.last = None
        self.adjusted = 0
        self.changed = False

    def estimate(self, mon, now):
        eng = mon.eng
        n_spk = len(eng.speakers)
        if n_spk < 2 or eng.t0 is None:
            return None
        L = mon.lat
        t_b = now - 0.2
        t_a = t_b - self.WIN_S
        mic = mon.mic_segment(t_a, t_b)
        if mic is None:
            return None
        if abs(mon.sr - SR) > 1:
            from scipy.signal import resample_poly
            mic = resample_poly(mic, int(SR), int(mon.sr)).astype(np.float32)
        f_a = int(round((t_a - L - eng.t0) * SR))
        ref = eng.ref_segment(f_a, f_a + len(mic))
        if ref is None or np.abs(ref).max() < 1e-4 or np.abs(mic).max() < 1e-4:
            return None                                          # silencio: nada que medir
        NF, hop = self.NF, self.NF // 2
        w = np.hanning(NF)
        nb = NF // 2 + 1
        Srr = np.zeros((nb, n_spk, n_spk), complex)
        Srm = np.zeros((nb, n_spk), complex)
        Smm = np.zeros(nb)
        frames = 0
        for k in range(0, len(mic) - NF, hop):
            M = np.fft.rfft(mic[k:k + NF] * w)
            R = np.fft.rfft(ref[k:k + NF].T * w, axis=1).T       # (nb, n_spk)
            Srr += R[:, :, None] * R[:, None, :].conj()
            Srm += R * M.conj()[:, None]
            Smm += np.abs(M) ** 2
            frames += 1
        if frames < 2 * n_spk:
            return None
        f = np.fft.rfftfreq(NF, 1 / SR)
        band = (f >= 300) & (f <= 6000)
        lam = 1e-3 * np.trace(Srr, axis1=1, axis2=2).real / n_spk
        H = np.zeros((nb, n_spk), complex)
        A = Srr[band] + lam[band, None, None] * np.eye(n_spk)
        H[band] = np.linalg.solve(A, Srm[band][..., None])[..., 0].conj()
        out = []
        lim = int(self.SEARCH_MS / 1000 * SR)
        for i in range(n_spk):
            Hi = H[:, i]
            mag = np.abs(Hi)
            if mag[band].max() == 0:
                out.append(None)
                continue
            phat = np.where(band, Hi / (mag + 1e-12), 0)
            h = np.fft.irfft(phat, NF)
            h = np.concatenate([h[-lim:], h[:lim + 1]])          # retrasos de −lim a +lim
            k = int(np.argmax(np.abs(h)))
            conf = float(np.abs(h[k]) / (np.median(np.abs(h)) + 1e-12))
            out.append((float((k - lim) / SR * 1000), conf))
        return out

    def step(self, mon, now, log=None):
        # Tras una corrección: esperar a que el ajuste suave termine (~15 s) y a una ventana nueva entera.
        if now < self.block_until:
            return
        # Solo medir con los buffers quietos: si alguno se realineó dentro de la ventana o está
        # lejos de su objetivo, lo que se mide es el bache y no el desfase real.
        eng = mon.eng
        if any(now - sp.snap_t < self.WIN_S + 1.0 for sp in eng.speakers):
            return
        if any(abs(st["fill_err_ms"]) > 6.0 for st in eng.status()):
            self.hist.clear()
            return
        est = self.estimate(mon, now)
        if not est or any(e is None or e[1] < 8 for e in est):
            return
        d = np.array([e[0] for e in est])
        rel = d - np.median(d)                                   # quién se adelanta (−) o atrasa (+)
        self.hist.append(rel)
        H = np.array(self.hist)
        med = np.median(H, axis=0)
        self.last = {"rel_ms": [round(float(x), 1) for x in med], "conf": [round(e[1]) for e in est]}
        # Corrige solo si 3 estimaciones seguidas coinciden (±1,5 ms) y el desfase vale la pena (>0,8 ms).
        if len(H) == 3 and np.std(H, axis=0).max() < 1.5 and np.abs(med).max() > 0.8:
            eng = mon.eng
            fix = np.clip(med, -8.0, 8.0)
            new = [sp.delay - int(round(f / 1000 * SR)) for sp, f in zip(eng.speakers, fix)]
            shift = -min(0, min(new))                            # nadie puede quedar con retraso negativo
            for sp, nd in zip(eng.speakers, new):
                sp.slew_delay(nd + shift)
            self.hist.clear()
            self.block_until = now + 15.0 + self.WIN_S
            self.last = {"rel_ms": [0.0] * len(fix), "conf": self.last["conf"], "fixed_ms": [round(float(x), 1) for x in fix]}
            self.adjusted += 1
            self.changed = True


# --- cancelador de zumbidos -------------------------------------------------------------
class HumTone:
    def __init__(self, f: float):
        self.f = f
        self.w = 0j          # amplitud compleja de la antionda (en la entrada del parlante)
        self.S = None        # cómo llega al micrófono (complejo): ganancia + fase + retraso
        self.noise = None    # ruido medido sin antionda
        self.resid = None
        self.state = "probe_off"
        self.best = None


class HumGen:
    """Suma Re(w·e^{j2πf(t-T0)}) en la señal del parlante elegido, con t el reloj de pared."""

    def __init__(self, speaker_idx: int, T0: float):
        self.idx, self.T0 = speaker_idx, T0
        self.tones: list = []

    def add(self, mix: np.ndarray, g0: int, t0: float):
        if not self.tones or t0 is None:
            return
        t = t0 + (g0 + np.arange(len(mix))) / SR - self.T0
        y = np.zeros(len(mix))
        for tn in list(self.tones):
            if tn.w:
                y += (tn.w * np.exp(2j * np.pi * tn.f * t)).real
        mix[:, self.idx] += np.clip(y, -0.5, 0.5).astype(np.float32)


class RoomMonitor:
    def __init__(self, engine, cfg: dict, mic=None):
        self.eng, self.cfg = engine, cfg
        self.dev = find_input_device(mic)
        self.sr = float(sd.query_devices(self.dev)["default_samplerate"])
        self.buf = np.zeros(int(self.sr * 12), np.float32)
        self.count = 0
        self.lock = threading.Lock()
        self.lat = float(cfg.get("acoustic_latency_ms") or 400.0) / 1000
        tfs = [s.get("tf_db") for s in engine.config["speakers"]]
        self.tf = None if any(t is None or len(t) != len(eqmod.THIRD) for t in tfs) else \
            np.array([10 ** (np.array(t) / 10) for t in tfs])       # (parlantes, bandas) ganancia de potencia
        self.history = deque(maxlen=600)       # niveles de ruido (dB) de los últimos ~5 min
        self.noise_db = None
        self.noise_curve = None
        self.baseline_db = None
        self.boost = 0.0
        self.stats = {}
        self.running = True
        self.anchors = deque(maxlen=400)       # ~10–35 s de callbacks
        self.mic_fit = None
        self.stream = sd.InputStream(device=self.dev, channels=1, samplerate=self.sr,
                                     dtype="float32", callback=self._cb)
        self.stream.start()
        self.T0 = time.monotonic()
        self.hum_gen = None
        self.hum_cycle_t = 0.0
        self.sync = SyncTracker()
        self.sync_t = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --- captura --------------------------------------------------------------------
    def _cb(self, indata, frames, t, status):
        x = indata[:, 0]
        with self.lock:
            i0 = self.count % len(self.buf)
            first = min(frames, len(self.buf) - i0)
            self.buf[i0:i0 + first] = x[:first]
            if first < frames:
                self.buf[:frames - first] = x[first:]
            # (hora, muestra): el reloj del micrófono no corre exacto a su frecuencia nominal y entrega
            # en bloques, así que se ajusta una recta y se recoloca. Si CoreAudio da la hora de captura
            # de la primera muestra del bloque (exacta), se usa esa; si no, la de llegada del callback.
            adc = adc_time(t)
            self.exact = adc is not None
            if adc is not None:
                self.anchors.append((adc, self.count))
            self.count += frames
            if adc is None and self.count > self.sr * 0.5:   # el arranque llega a ráfagas: no sirve
                self.anchors.append((time.monotonic(), self.count))
            if self.anchors and ((self.count // frames) % 8 == 0 or self.mic_fit is None):
                self._fit_clock()

    def _fit_clock(self):
        a = np.array(self.anchors, dtype=float)
        t, n = a[:, 0], a[:, 1]
        if len(a) >= 8 and n[-1] - n[0] > self.sr:
            b, c = np.polyfit(n - n[0], t, 1)            # s por muestra
            b = float(np.clip(b, 0.99 / self.sr, 1.01 / self.sr))
        else:
            b = 1.0 / self.sr
        r = (t - b * n)[-48:]                            # solo lo reciente (la tasa varía un poco)
        # Hora de captura: exacta, basta la mediana. Hora de llegada: los callbacks llegan tarde, nunca
        # antes, así que la envolvente inferior es la hora real.
        off = float(np.median(r) if getattr(self, "exact", False) else np.min(r))
        self.mic_fit = (off, b)

    def mic_index(self, t: float) -> int:
        off, b = self.mic_fit
        return int(round((t - off) / b))

    def mic_segment(self, t_a: float, t_b: float):
        """Micrófono entre dos instantes de reloj de pared (monotonic)."""
        if self.mic_fit is None:
            return None
        i_a, i_b = self.mic_index(t_a), self.mic_index(t_b)
        with self.lock:
            if i_a < max(0, self.count - len(self.buf)) or i_b > self.count or i_b <= i_a:
                return None
            idx = np.arange(i_a, i_b) % len(self.buf)
            return self.buf[idx].copy()

    def stop(self):
        self.running = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.eng.masker = None
        self.eng.hum = None
        self.eng.extra_db = self.eng.center_extra_db = 0.0

    # --- estimación de ruido ------------------------------------------------------------
    def _estimate_noise(self, now: float):
        win = 1.0
        t_b = now - 0.15
        t_a = t_b - win
        mic = self.mic_segment(t_a, t_b)
        if mic is None or self.eng.t0 is None:
            return
        mic_e = _band_energy(mic, self.sr, NOISE_BANDS) / win
        pred = np.zeros(len(NOISE_BANDS))
        if self.tf is not None:
            f_a = int((t_a - self.lat - self.eng.t0) * SR)
            ref = self.eng.ref_segment(f_a, f_a + int(win * SR))
            if ref is None:
                return
            for i in range(ref.shape[1]):
                pred += _band_energy(ref[:, i], SR, NOISE_BANDS) / win * self.tf[i, _SEL]
        # Solo cuentan las bandas donde el micrófono oye claramente más que lo nuestro (margen 3 dB).
        noise = np.where(mic_e > pred * 2.0, mic_e - pred * 2.0, 0.0)
        # Si nuestra reproducción tapa todo, el ruido "no se ve": cuenta como muy bajo (−30 dB bajo lo que
        # oye el micrófono), no como "sin dato", para que el fondo habitual se aprenda igual.
        noise_db = float(10 * np.log10(noise.sum() + mic_e.sum() * 1e-3))
        self.noise_db = noise_db if self.noise_db is None else 0.7 * self.noise_db + 0.3 * noise_db
        self.history.append(self.noise_db)
        self.baseline_db = float(np.percentile(self.history, 10))
        curve = 10 * np.log10(np.maximum(mic_e - pred, 1e-20))
        self.noise_curve = curve if self.noise_curve is None else 0.8 * self.noise_curve + 0.2 * curve

    # --- volumen adaptativo ---------------------------------------------------------------
    def _adapt(self, dt: float, snd: dict):
        if not snd.get("adaptive") or self.noise_db is None or len(self.history) < 6:
            target = 0.0
        else:
            target = float(np.clip((self.noise_db - self.baseline_db - 4.0) * 0.6, 0.0, 9.0))
        step = (1.5 if target > self.boost else 0.6) * dt     # sube rápido, baja suave
        self.boost += float(np.clip(target - self.boost, -step, step))
        self.eng.extra_db = self.boost * 0.6
        self.eng.center_extra_db = self.boost * 0.5          # los diálogos suben un poco más

    # --- enmascaramiento ---------------------------------------------------------------
    def _mask(self, snd: dict, now: float):
        if not snd.get("mask"):
            self.eng.masker = None
            return
        if self.eng.masker is None:
            self.eng.masker = Masker()
            self.mask_t = 0.0
        m = self.eng.masker
        if self.noise_curve is None or self.tf is None:
            m.shape([0.0] * len(eqmod.THIRD), 10 ** ((-42 + snd.get("mask_db", 0)) / 20))   # rosa suave
            return
        if now - getattr(self, "mask_t", 0.0) < 3.0:      # la forma cambia lento: sin saltos audibles
            return
        self.mask_t = now
        # Energía que hay que emitir por banda para que en el micrófono la máscara iguale al ruido.
        need = 10 ** (self.noise_curve / 10) / (self.tf.sum(axis=0)[_SEL] + 1e-20)
        need_db = 10 * np.log10(need)
        # Forma del filtro: lo necesario, compensando que las bandas de 1/3 de octava se ensanchan con f.
        shape_sel = need_db - 10 * np.log10(_T[_SEL])
        shape = np.interp(np.log2(_T), np.log2(_T[_SEL]), shape_sel - shape_sel.max())
        m.shape(shape, 1.0)
        # Ganancia calibrada con un segundo de prueba del propio filtro.
        from scipy.signal import oaconvolve
        test = oaconvolve(np.random.default_rng(0).standard_normal(SR), m.h)[:SR]
        got = _band_energy(test, SR, NOISE_BANDS)
        g2 = float(np.median(need / got)) * 10 ** (snd.get("mask_db", 0.0) / 10)
        m.gain = float(min(0.3, np.sqrt(g2)))

    # --- cancelador de zumbidos ------------------------------------------------------------
    def _detect_hum(self, now: float) -> list:
        mic = self.mic_segment(now - 8.15, now - 0.15)   # 8 s: frecuencia con ~0,01 Hz de precisión
        if mic is None:
            return []
        n = len(mic)
        X = np.abs(np.fft.rfft(mic * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1 / self.sr)
        sel = np.where((f >= 40) & (f <= 200))[0]
        found = []
        for i in sel[2:-2]:
            if X[i] < X[i - 1] or X[i] < X[i + 1]:
                continue
            neigh = np.median(X[max(0, i - 40):i + 40])
            if X[i] > neigh * 10 ** (15 / 20):            # 15 dB sobre lo que lo rodea
                a, b, c = np.log(X[i - 1]), np.log(X[i]), np.log(X[i + 1])
                p = 0.5 * (a - c) / (a - 2 * b + c)
                found.append((float(f[i] + p * (f[1] - f[0])), float(X[i])))
        found.sort(key=lambda x: -x[1])
        if not found:
            return []
        top = found[0][1]
        floor = 1e-4 * n / 2          # ~−80 dBFS: por debajo no vale la pena (ni se oye)
        return [fr for fr, a in found[:3] if a >= top * 0.1 and a >= floor]

    def _demod(self, f: float, t_a: float, t_b: float):
        x = self.mic_segment(t_a, t_b)
        if x is None:
            return None
        t = t_a + np.arange(len(x)) / self.sr - self.T0
        return 2 * np.mean(x * np.exp(-2j * np.pi * f * t))

    def _hum(self, now: float, snd: dict):
        if not snd.get("hum"):
            self.eng.hum = None
            self.hum_gen = None
            self.stats.pop("hum", None)
            return
        if self.hum_gen is None:
            from . import dsp
            bass = dsp.bass_speakers(self.eng.config["speakers"])
            self.hum_gen = HumGen(bass[0] if bass else 0, self.T0)
            self.hum_cycle_t = now
            self.eng.hum = self.hum_gen
        g = self.hum_gen
        if now - self.hum_cycle_t < 1.9:       # cada ciclo: ~0,9 s para que llegue el cambio + 1 s de medida
            return
        self.hum_cycle_t = now
        if not g.tones:
            for fr in self._detect_hum(now):
                if all(abs(fr - b) > 0.5 for b in getattr(g, "blocked", [])):
                    g.tones.append(HumTone(fr))
            if not g.tones:
                self.stats["hum"] = "sin zumbidos estables"
            return
        t_b = now - 0.1
        t_a = t_b - 1.0
        out = []
        for tn in g.tones:
            Z = self._demod(tn.f, t_a, t_b)
            if Z is None:
                continue
            if tn.state == "probe_off":            # 1) ruido solo
                tn.noise = Z
                guess = np.sqrt(self.tf[g.idx, eqmod.THIRD.index(min(eqmod.THIRD, key=lambda c: abs(c - tn.f)))]) \
                    if self.tf is not None else 0.05
                tn.w = abs(Z) / max(guess, 1e-4) * 0.5 + 0j      # 2) sonda conocida
                tn.state = "probe_on"
            elif tn.state == "probe_on":
                tn.S = (Z - tn.noise) / tn.w                    # cómo llega la antionda al micrófono
                tn.w = -tn.noise / tn.S if abs(tn.S) > 1e-9 else 0j
                tn.state = "cancel"
            else:                                               # 3) ajuste fino en ciclos
                red = 20 * np.log10(abs(Z) / max(abs(tn.noise), 1e-12))
                tn.bad = getattr(tn, "bad", 0) + 1 if red > 1.5 else 0
                if tn.bad >= 2:                                 # empeora dos ciclos seguidos: re-medir
                    tn.w, tn.state, tn.bad = 0j, "probe_off", 0
                    continue
                # Refinar la llegada con lo que cambió en el micrófono al cambiar la antionda.
                prev = getattr(tn, "prev", None)
                if prev is not None and abs(tn.w - prev[0]) > 0.1 * abs(tn.w) + 1e-9:
                    S_obs = (Z - prev[1]) / (tn.w - prev[0])
                    if 0.3 < abs(S_obs / tn.S) < 3:
                        tn.S = 0.7 * tn.S + 0.3 * S_obs
                tn.prev = (tn.w, Z)
                tn.hist = (getattr(tn, "hist", []) + [red])[-15:]
                new_w = tn.w - 0.5 * Z / tn.S
                tn.w = min(abs(new_w), 0.5) * np.exp(1j * np.angle(new_w))
                out.append(f"{tn.f:.1f} Hz {red:+.0f} dB")
        if out:
            self.stats["hum"] = ", ".join(out)
        # Protección: tras ~30 s, si el promedio no baja al menos 3 dB, la antionda no sirve aquí.
        for tn in list(g.tones):
            h_ = getattr(tn, "hist", [])
            if len(h_) >= 15 and np.mean(h_) > -3:
                g.tones.remove(tn)
                g.blocked = getattr(g, "blocked", []) + [tn.f]
                self.stats["hum_off"] = (f"{tn.f:.0f} Hz: no se pudo cancelar (el retraso del audio varía "
                                         f"demasiado para la fase que exige); se apagó solo.")

    # --- bucle ---------------------------------------------------------------------------
    def _loop(self):
        from . import dsp
        last = time.monotonic()
        time.sleep(1.5)
        while self.running:
            now = time.monotonic()
            dt, last = now - last, now
            snd = dsp.sound_cfg(self.eng.config)
            try:
                self._estimate_noise(now)
                self._adapt(dt, snd)
                self._mask(snd, now)
                self._hum(now, snd)
                if snd.get("autosync") and now - self.sync_t > 3.0 and now - self.T0 > 6.0:
                    self.sync_t = now
                    self.sync.step(self, now)
            except Exception as e:      # el monitor nunca debe tumbar la reproducción
                self.stats["error"] = str(e)
            self.stats.update({
                "noise_db": None if self.noise_db is None else round(self.noise_db, 1),
                "baseline_db": None if self.baseline_db is None else round(self.baseline_db, 1),
                "boost_db": round(self.boost, 1),
                "calibrated": self.tf is not None,
                "sync": self.sync.last, "sync_adjusted": self.sync.adjusted,
            })
            time.sleep(0.5)
