"""Motor multi-parlante: una salida independiente por parlante Bluetooth, con retraso, ganancia
y corrección continua de deriva de reloj (remuestreo adaptativo por parlante)."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.signal import oaconvolve, sosfilt

from . import dsp
from . import eq as eqmod
from . import render
from . import spatial3d
from .clock import dac_time
from .fifo import Fifo
from .sources import SR

BLOCK = 480                  # productor; distinto de OUT_BLOCK a propósito (ver Speaker._level)
BASE_FILL = int(0.06 * SR)   # colchón por parlante (además del retraso Bluetooth propio)
KEEPALIVE = 10 ** (-50 / 20)  # 18 Hz a −50 dBFS: inaudible, pero el parlante «ve» señal y no se duerme
# Control por retraso real: la hora en que va a sonar cada bloque (la informa CoreAudio, ±10 µs) menos la
# hora en que se produjo esa muestra. Medido con parlantes BT reales abiertos a 48 kHz (macOS convierte a
# 44,1): la hora de llegada de las llamadas deambula ±3–7 ms, así que el control por nivel de buffer veía
# errores de 10–30 ms que no existían y, al perseguirlos (hasta 3000 ppm), los creaba de verdad. La versión
# anterior de este control usaba la hora de llegada de la llamada y por eso salió peor (0,35–3 ms).
# Apagado (2026-10-04): en pruebas silenciosas daba ±0,007 ms, pero dentro de la app (en vivo y al calibrar)
# hizo cientos de saltos (JBL 216, Bose 362) y la calibración quedó en ±40 ms. Hasta entender por qué, por nivel.
USE_LAT = False
DEV_LAT_CALLBACKS = 200      # ~2 s para fijar la latencia propia del dispositivo (mediana, reproducible)
DEV_LAT_CHANGE = 0.005       # s: cambio de la latencia propia del dispositivo que se considera real
BAD_TS_MAX = 10              # de las últimas 200 horas de reproducción: más incoherentes → por nivel
LAT_JUMP = 0.004             # s: un error mayor (corte del Bluetooth, bache de macOS) se corrige de un salto
LAT_JUMP_CALLBACKS = 3       # …si se sostiene ~30 ms (no por una medición suelta)
NATIVE_RATE = False          # abrir cada parlante a su frecuencia propia (experimental)
# Búfer de CoreAudio por parlante. Con «low» (~11 ms) Python a veces no llegaba a tiempo (el GIL lo
# comparten el productor y las salidas) y macOS se saltaba un bloque entero: trabones y desfase de 11,6 ms.
# El retraso extra no afecta la sincronía: el control usa la hora real de reproducción.
OUT_LATENCY = 0.1
OUT_BLOCK = 512              # cuadros por llamada de salida (0 = lo que decida macOS)
LAT_KP = 0.2                 # control por retraso real: 1/s (constante de tiempo ~5 s)
LAT_KI = 0.01                # integral (absorbe la diferencia fija de reloj)
LAT_ALPHA = 0.03             # suavizado de la medición por callback (~0,4 s)
KP = 2e-6                    # control proporcional de deriva
KI = 5e-8                    # control integral de deriva
MAX_PPM = 3000
LEVEL_BY_TIME = False       # por tiempo: mejor en teoría, pero macOS entrega callbacks en ráfagas y lo sesga
INTEG_LIMIT = int(0.004 * SR)   # tope del error que entra al integrador (los baches no lo enrollan)
FAST_GAIN = 4.0              # ganancia extra al arrancar (KP×4, KI×16)
FAST_CALLBACKS = 500        # ~5 s con bloques de 512
HARD_RESYNC = int(0.06 * SR)


import sys as _sys
_sys.setswitchinterval(0.001)   # el hilo de audio espera menos por el GIL (por defecto 5 ms)

_ACTIVE = 0   # motores con flujos abiertos en este proceso


def refresh_devices():
    """Vuelve a leer la lista de dispositivos de macOS. Hace falta cuando un parlante Bluetooth se
    reconecta: cambia de identificador y la lista vieja apunta a algo que ya no existe. Solo es seguro
    sin flujos abiertos, así que se hace únicamente cuando no hay ningún motor sonando."""
    if _ACTIVE == 0:
        sd._terminate()
        sd._initialize()


def find_output_device(pattern) -> int:
    if isinstance(pattern, int):
        return pattern
    pat = str(pattern).lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and pat in d["name"].lower():
            return i
    raise RuntimeError(f"No encontré parlante/salida que contenga '{pattern}'. "
                       f"Corre `python3 -m sbt devices` y revisa que esté conectado.")


def soft_limit(x: np.ndarray, t: float = 0.75) -> np.ndarray:
    """Lineal hasta t; por encima comprime suavemente hacia 1.0 (sin el chasquido del recorte duro)."""
    a = np.abs(x)
    if a.max(initial=0.0) <= t:
        return x
    over = a > t
    y = x.copy()
    y[over] = np.sign(x[over]) * (t + (1 - t) * np.tanh((a[over] - t) / (1 - t)))
    return y


class Speaker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg["name"]
        self.device = find_output_device(cfg["device"])
        info = sd.query_devices(self.device)
        self.channels = min(2, int(info["max_output_channels"]))
        # Se abre a la frecuencia propia del parlante (los BT suelen ir a 44,1 kHz): así macOS no convierte
        # por dentro (lo que entrega las llamadas a tirones) y la conversión la hace el remuestreador sinc.
        self.dev_sr = (float(info["default_samplerate"]) or float(SR)) if NATIVE_RATE else float(SR)
        self.base_ratio = SR / self.dev_sr
        self.delay = int(round(cfg.get("delay_ms", 0.0) * SR / 1000))
        # ganancia = la de la calibración (iguala parlantes) + el ajuste del usuario (encima)
        self.cal_gain_db = float(cfg.get("gain_db", 0.0))
        self.trim_db = float(cfg.get("trim_db", 0.0))
        self.gain = 10 ** ((self.cal_gain_db + self.trim_db) / 20)
        self.mute = False
        self.ka_phase = 0.0
        self.eq = None              # (secciones, estado) del ecualizador del usuario
        self.fir = eqmod.fir([0.0] * len(eqmod.THIRD))   # igualación automática (siempre activa:
        self.fir_tail = np.zeros(len(self.fir) - 1)      # así el retraso es el mismo con o sin ella)
        # Protección de parlantes pequeños: lo que no pueden reproducir solo los hace distorsionar.
        ph = eqmod.protect_hz(cfg)
        self.protect = dsp.lr4("highpass", ph, 1) if ph else None
        self.protect_hz = ph
        self.bass_lim = dsp.BassLimiter()
        self.bass_lim_blocks = 0
        self.limit_blocks = 0       # bloques donde actuó el limitador (saturación)
        self.blocks = 0
        self.fifo = Fifo(SR * 4)
        self.target = BASE_FILL + self.delay
        self.ema = float(self.target)
        self.integ = 0.0
        self.integ_good = 0.0
        self.calm = 0
        self.snap_t = 0.0
        self.ratio = self.base_ratio
        self.resyncs = 0
        self.stream = None
        self.warm = 0
        self.snap_at = 48          # tras ~0,5 s (varios ciclos del productor) alinea al objetivo
        self.pending_delay = None
        # Control por retraso real (estilo JACK/PipeWire): hora de reproducción − hora de producción.
        self.eng = None
        self.offset = 0            # índice del fifo que corresponde al cuadro producido 0
        self.clock = time.monotonic
        self.lat_ema = None
        self.lat_set = None        # retraso objetivo (s); se fija tras alinear
        self.lat_integ = 0.0
        self.lat_err = 0.0
        self.dev_lats = []         # hora de sonar − hora de la llamada, al arrancar
        self.dev_lat = None        # su mediana: lo que el dispositivo agrega (no cambia entre aperturas)
        self.big = 0               # callbacks seguidos con error grande
        self.prev_dac = None
        self.prev_frames = 0
        self.bad_ts = deque(maxlen=200)   # horas de reproducción incoherentes, recientes
        self.jumps = 0

    def _level(self, frames: int) -> float:
        """Llenado del buffer en el punto medio de esta lectura.

        Por tiempo (por defecto): cuánto «debería» haber escrito el productor a esta hora según su
        línea de tiempo continua, menos lo leído. Contar lo escrito de verdad sale escalonado: el
        productor escribe de a 512 y la salida lee de a 512 casi al mismo ritmo, así que una deriva
        real queda invisible hasta que la fase cruza un bloque y aparece como un salto de ~10 ms."""
        f = self.fifo
        e = self.eng
        if LEVEL_BY_TIME and e is not None and e.prod_fit is not None:
            a, b = e.prod_fit
            w_now = self.offset + (self.clock() - a) / b
            return w_now - (f.r + f.frac) + f.pad - 0.5 * frames * self.ratio
        return f.fill() - 0.5 * frames * self.ratio

    def _snap(self):
        """Alinea el buffer al objetivo usando el nivel PROMEDIADO: el instantáneo es un diente de sierra
        de ±1 bloque del productor (~10 ms) y alinear con él dejaba errores de hasta 10 ms."""
        e = self.ema - self.target
        if e > 0:
            self.fifo.drop(int(round(e)))
        elif e < 0:
            self.fifo.add_pad(int(round(-e)))
        self.ema = float(self.target)
        # El integrador guarda la deriva aprendida entre los relojes: no se borra al realinear.
        self.snap_t = time.monotonic()
        self.lat_set = None
        self.lat_ema = None
        self.lat_integ = 0.0
        self.set_wait = 40         # callbacks a esperar antes de fijar el objetivo nuevo

    def _shift(self, new_delay: int):
        """Cambio de retraso exacto: rellena o descarta justo la diferencia (relativo, sin re-medir)."""
        d = int(new_delay) - self.delay
        if d > 0:
            self.fifo.add_pad(d)
        elif d < 0:
            self.fifo.drop(-d)
        self.delay = int(new_delay)
        self.target = BASE_FILL + self.delay
        self.ema += d
        if self.lat_set is not None:
            self.lat_set += d / SR
            self.lat_ema += d / SR
        self.snap_t = time.monotonic()

    def _latency(self, t_play: float) -> float | None:
        """Retraso real de la muestra que va a sonar primero: cuándo suena − cuándo se produjo (s)."""
        e = self.eng
        if e is None or e.prod_fit is None:
            return None
        f = self.fifo
        idx = (f.r + f.frac - f.pad) - self.offset          # los ceros pendientes suenan antes que f.r
        a, b = e.prod_fit
        return t_play - (a + b * idx)

    def _lat_target(self) -> float:
        return (BASE_FILL + self.delay) / SR + self.dev_lat

    def callback(self, outdata, frames, t, status):
        self.warm += 1
        if self.pending_delay is not None:
            nd, self.pending_delay = self.pending_delay, None
            if self.warm > self.snap_at:
                self._shift(nd)
            else:
                self.delay = nd
                self.target = BASE_FILL + nd
        lv = self._level(frames)
        if self.warm <= self.snap_at:
            # Arranque: promedio simple de varios ciclos del productor, luego alinear una vez.
            self.ema = lv if self.warm == 1 else self.ema + (lv - self.ema) / self.warm
            if self.warm == self.snap_at:
                self._snap()
        else:
            self.ema += 0.02 * (lv - self.ema)
        err = self.ema - self.target
        t_play = dac_time(t) if USE_LAT else None
        if t_play is not None:
            # Las horas deben avanzar justo lo que duró el bloque anterior. Si fallan seguido (BlackHole
            # con otra app usando bloques grandes las da corridas), no se confía en ellas: control por nivel.
            dac = t.outputBufferDacTime
            if self.prev_dac is not None:
                self.bad_ts.append(abs(dac - self.prev_dac - self.prev_frames / self.dev_sr) > 0.0005)
            self.prev_dac, self.prev_frames = dac, frames
            if sum(self.bad_ts) > BAD_TS_MAX:
                t_play = None
        lat = self._latency(t_play) if (t_play is not None and self.warm > self.snap_at) else None
        if lat is not None:
            self.dev_lats.append(t.outputBufferDacTime - t.currentTime)
            if len(self.dev_lats) > DEV_LAT_CALLBACKS:
                del self.dev_lats[0]
            if self.dev_lat is None:
                if len(self.dev_lats) >= DEV_LAT_CALLBACKS:
                    self.dev_lat = float(np.median(self.dev_lats))
            elif self.warm % 16 == 0:
                # Si el dispositivo cambia su propio buffer (otra app lo abre con bloques grandes, el
                # Bluetooth renegocia), cambia su latencia: el objetivo la sigue o el buffer queda corto.
                m = float(np.median(self.dev_lats[-64:]))
                if abs(m - self.dev_lat) > DEV_LAT_CHANGE:
                    self.dev_lat = m
                    if self.lat_set is not None:
                        self.lat_set = self._lat_target()
            # Un valor lejos del objetivo es un bache (lo resuelve el salto): no entra al promedio.
            if self.lat_ema is None:
                self.lat_ema = lat
            elif self.lat_set is None or abs(lat - self.lat_set) <= LAT_JUMP:
                self.lat_ema += LAT_ALPHA * (lat - self.lat_ema)
            if self.lat_set is None and self.dev_lat is not None:
                self.set_wait = getattr(self, "set_wait", 40) - 1
                if self.set_wait <= 0:
                    # Objetivo absoluto (colchón + retraso + latencia propia): igual en cada arranque.
                    # Se alinea una vez, exacto, y desde ahí solo se sigue la deriva.
                    self.lat_set = self._lat_target()
                    e = int(round((self.lat_ema - self.lat_set) * SR))
                    if e > 0:
                        self.fifo.drop(e)
                    elif e < 0:
                        self.fifo.add_pad(-e)
                    self.ema -= e
                    self.lat_ema = self.lat_set
                    # La deriva se aprende de cero: la del arranque por nivel viene sesgada (medido: a veces
                    # 3000 ppm que luego tardaban un minuto en deshacerse).
                    self.snap_t = time.monotonic()
        if self.warm <= self.snap_at:
            self.ratio = self.base_ratio * (1.0 + KI * self.integ)   # sin control hasta alinear
        elif self.warm > self.snap_at + 50 and (abs(self.lat_err) * SR > HARD_RESYNC if self.lat_set is not None
                                                  else abs(err) > HARD_RESYNC):
            lat_integ = self.lat_integ
            self._snap()
            self.integ = self.integ_good
            self.lat_integ = lat_integ
            self.lat_err = 0.0
            self.resyncs += 1
        elif self.lat_set is not None:
            # Control por retraso real: sin el ruido de las ráfagas del productor.
            e_now = lat - self.lat_set if lat is not None else 0.0
            self.big = self.big + 1 if abs(e_now) > LAT_JUMP else 0
            if self.big >= LAT_JUMP_CALLBACKS:
                # La medición es exacta (±10 µs): mejor un salto justo ahora que segundos de tono corrido.
                n = int(round(e_now * SR))
                if n > 0:
                    self.fifo.drop(n)
                else:
                    self.fifo.add_pad(-n)
                self.ema -= n
                self.lat_ema = lat - n / SR                 # lo que queda tras el salto (≈ objetivo)
                self.big = 0
                self.jumps += 1
            e = self.lat_ema - self.lat_set
            self.lat_err = e
            dt = frames / SR
            self.lat_integ = float(np.clip(self.lat_integ + e * dt, -MAX_PPM * 1e-6 / LAT_KI, MAX_PPM * 1e-6 / LAT_KI))
            adj = LAT_KP * e + LAT_KI * self.lat_integ
            self.ratio = self.base_ratio * (1.0 + float(np.clip(adj, -MAX_PPM * 1e-6, MAX_PPM * 1e-6)))
        else:
            # Los primeros segundos, control más rápido: aprende la deriva de los relojes en ~3 s
            # en vez de ~20 s (luego vuelve a lo suave para no modular el tono).
            g = FAST_GAIN if self.warm < self.snap_at + FAST_CALLBACKS else 1.0
            # Un error grande suele ser un bache (macOS pausó la salida, se abrió otro dispositivo…):
            # al integrador entra recortado, así aprende derivas grandes sin «enrollarse» con baches.
            self.integ += g * g * float(np.clip(err, -INTEG_LIMIT, INTEG_LIMIT)) * frames / SR
            self.integ = float(np.clip(self.integ, -MAX_PPM * 1e-6 / KI, MAX_PPM * 1e-6 / KI))
            if abs(err) < INTEG_LIMIT:
                self.calm += 1
                if self.calm % 200 == 0:
                    self.integ_good = self.integ          # deriva aprendida en calma
            else:
                self.calm = 0
            adj = g * KP * err + KI * self.integ
            self.ratio = self.base_ratio * (1.0 + float(np.clip(adj, -MAX_PPM * 1e-6, MAX_PPM * 1e-6)))
        mono = self.fifo.read(frames, self.ratio)
        if self.mute:
            mono[:] = 0
        ph = self.ka_phase + 2 * np.pi * 18.0 / self.dev_sr * np.arange(frames)
        mono += (KEEPALIVE * np.sin(ph)).astype(np.float32)
        self.ka_phase = float((ph[-1] + 2 * np.pi * 18.0 / self.dev_sr) % (2 * np.pi))
        outdata[:] = mono[:, None]

    def set_auto(self, curve_db):
        self.fir = eqmod.fir(curve_db)

    def set_eq(self, gains_db):
        sections = eqmod.sos(gains_db)
        self.eq = None if sections is None else (sections, np.zeros((len(sections), 2)))

    def set_delay(self, samples: int):
        # Se aplica dentro del callback de audio para no competir con él.
        self.pending_delay = max(0, int(samples))
        if self.stream is None or not self.stream.active:
            self.delay = self.pending_delay
            self.target = BASE_FILL + self.delay
            self.pending_delay = None

    def slew_delay(self, samples: int):
        """Cambia el retraso sin saltos: el control de deriva lleva el buffer al nuevo objetivo poco a
        poco (variación de velocidad imperceptible), en vez de cortar o rellenar de golpe."""
        samples = max(0, int(samples))
        if self.lat_set is not None:
            self.lat_set += (samples - self.delay) / SR      # = _lat_target() con el retraso nuevo
        self.delay = samples
        self.target = BASE_FILL + samples

    @property
    def settled(self) -> bool:
        return self.warm > self.snap_at and self.pending_delay is None


class Engine:
    def __init__(self, config: dict, source, master_db: float = 0.0):
        # En vivo la entrada (BlackHole) ya está abierta: refrescar la lista ahora la cerraría.
        # Para ese caso el servidor refresca antes de abrirla (check_live_output).
        if not getattr(source, "realtime", False):
            refresh_devices()
        self.config = config
        self.speakers = [Speaker(s) for s in config["speakers"]]
        self.source = source
        self.master_db = master_db
        self.master = 10 ** (master_db / 20)
        self.sound = dsp.SoundProcessor(dsp.sound_cfg(config), len(self.speakers),
                                        dsp.bass_speakers([s.cfg for s in self.speakers]))
        self.matrix = self._matrix(config["speakers"], config["listener"])
        self.injections = []       # (idx_parlante, frame_inicio, señal)
        self.inj_lock = threading.Lock()
        self.produced = 0
        self.t0 = None
        self.running = False
        self.thread = None
        self.peak = np.zeros(len(self.speakers))
        for i, j in eqmod.find_pairs([sp.cfg for sp in self.speakers], config["listener"]):
            fs = [f for f in (self.speakers[i].protect_hz, self.speakers[j].protect_hz) if f]
            if fs:                                   # el corte del más limitado, para los dos
                for k in (i, j):
                    self.speakers[k].protect_hz = max(fs)
                    self.speakers[k].protect = dsp.lr4("highpass", max(fs), 1)
        self.headroom = 1.0
        self.in_peak = 0.0
        self.paused = False
        self.ended = False
        self.prod_hist = []        # (cuadro producido, hora) para la regresión
        self.prod_fit = None       # (a, b): hora ≈ a + b·cuadro
        # Para el monitor del cuarto: copia de lo que se manda a cada parlante (antes de su ganancia),
        # indexada por cuadro, para restarlo de lo que oye el micrófono.
        self.ref_cap = SR * 12
        self.ref = np.zeros((self.ref_cap, len(self.speakers)), np.float32)
        self.extra_db = 0.0          # volumen adaptativo al ruido (general)
        self.center_extra_db = 0.0   # … y extra para diálogos
        self.masker = None           # generador de enmascaramiento (sbt/monitor.py)
        self.hum = None              # generador de antiruido para zumbidos (sbt/monitor.py)
        self.apply_eq(config)

    def _matrix(self, speakers_cfg, listener):
        snd = dsp.sound_cfg(self.config)
        if snd.get("party"):                       # fiesta: todo por todos (graves aparte, por el gestor)
            n = len(speakers_cfg)
            m = np.full((len(render.CHANNELS), n), 0.6 / np.sqrt(max(1, n)), np.float32)
            m[render.CHANNELS.index("C"), :] = 0.85 / np.sqrt(max(1, n))
            if self.sound.managing_bass:
                m[render.CHANNELS.index("LFE"), :] = 0
            return m
        m = render.build_matrix(speakers_cfg, listener, dsp.spatial(snd)["width"])
        if self.sound.managing_bass:
            m[render.CHANNELS.index("LFE"), :] = 0   # el LFE lo lleva el gestor de graves
        return m

    def set_master(self, db: float):
        self.master_db = float(db)
        self.master = 10 ** (self.master_db / 20)
        self.apply_eq(self.config)               # el dynamic EQ depende del volumen

    def set_sound(self, sound: dict):
        self.config["sound"] = dict(sound)
        self.sound = dsp.SoundProcessor(dsp.sound_cfg(self.config), len(self.speakers),
                                        dsp.bass_speakers([s.cfg for s in self.speakers]))
        self.matrix = self._matrix([s.cfg for s in self.speakers], self.config["listener"])
        self.apply_spatial(self.source)
        self.apply_eq(self.config)

    def apply_spatial(self, src):
        up = getattr(src, "upmix", None)
        if up is not None and hasattr(up, "front_amb"):
            sp = dsp.spatial(dsp.sound_cfg(self.config))
            up.surround, up.front_amb = sp["surround"], sp["front_amb"]
            if hasattr(up, "set_rear_delay"):
                # Los traseros llevan el ambiente retrasado (efecto Haas). Si caen en los mismos parlantes
                # que los delanteros (no hay parlantes atrás de ese lado), ese ambiente suena dos veces en el
                # mismo parlante, una 12–17 ms después: filtro peine (medido: ±3,6 dB cada ~80 Hz, «hueco»).
                m = self.matrix
                ci = render.CHANNELS.index
                def shared(a, b):
                    return float(np.dot(m[ci(a)], m[ci(b)]) / (np.linalg.norm(m[ci(a)]) * np.linalg.norm(m[ci(b)]) + 1e-9))
                up.set_rear_delay(shared("L", "Ls") < 0.5, shared("R", "Rs") < 0.5)

    def apply_eq(self, config: dict):
        """Ecualización por parlante = la del usuario (igual para todos) + la automática (propia)."""
        user = eqmod.user_bands(config)
        tilt = dsp.FLOOR_TILT.get(config.get("decor", {}).get("floor", "madera"), [0] * 8)
        user = [u + t for u, t in zip(user, tilt)]
        if dsp.sound_cfg(config)["loudness"]:
            user = [u + l for u, l in zip(user, dsp.loudness_bands(getattr(self, "master_db", 0.0)))]
        autos = [eqmod.auto_for(s.cfg, config) for s in self.speakers]
        top = max(user + [0.0]) + max([max(a) for a in autos] + [0.0])
        # Mismo margen para todos (no rompe la igualación); el limitador suave cubre el resto.
        self.headroom = 10 ** (-min(top, 3.0) / 20)
        for s, a in zip(self.speakers, autos):
            s.set_eq(user)
            s.set_auto(a)

    # --- control -----------------------------------------------------------
    def start(self):
        for s in self.speakers:
            s.fifo.write(np.zeros(s.target, dtype=np.float32))
            s.ema = float(s.target)
            s.eng = self
            s.offset = s.fifo.w
            try:
                s.stream = sd.OutputStream(device=s.device, channels=s.channels, samplerate=s.dev_sr,
                                           dtype="float32", latency=OUT_LATENCY,
                                           blocksize=int(round(OUT_BLOCK / s.base_ratio)) if OUT_BLOCK else 0,
                                           callback=s.callback)
            except Exception as e:
                for o in self.speakers:
                    if o.stream is not None:
                        o.stream.close()
                        o.stream = None
                raise RuntimeError(f"No pude abrir «{s.name}». Revisa que esté encendido y conectado por "
                                   f"Bluetooth (si lo reconectaste, vuelve a intentarlo). Detalle: {e}") from e
        global _ACTIVE
        _ACTIVE += 1
        self._counted = True
        self.running = True
        self.t0 = time.monotonic()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()
        for s in self.speakers:
            s.stream.start()

    def _track_production(self, frame: int):
        """Hora en que se produce cada cuadro. Con archivo/prueba el ritmo lo pone el reloj de pared,
        así que la ley es exacta: t0 + cuadro/SR. En vivo lo pone la entrada: se estima la velocidad
        por regresión (~20 s) manteniendo la referencia continua (sin saltos que el control persiga)."""
        if not getattr(self.source, "realtime", False):
            self.prod_fit = (self.t0, 1.0 / SR)
            self.prod_hist.clear()
            return
        now = time.monotonic()
        self.prod_hist.append((frame, now))
        if len(self.prod_hist) > 2000:
            del self.prod_hist[:len(self.prod_hist) - 2000]
        if len(self.prod_hist) >= 200 and len(self.prod_hist) % 50 == 0:
            h = np.array(self.prod_hist, dtype=np.float64)
            b_new = float(np.polyfit(h[:, 0] - h[0, 0], h[:, 1], 1)[0])
            if self.prod_fit is None:
                self.prod_fit = (h[-1, 1] - b_new * frame, b_new)
            else:
                a, b = self.prod_fit
                b = b + 0.2 * (b_new - b)                    # la velocidad cambia de a poco
                self.prod_fit = (a + (self.prod_fit[1] - b) * frame, b)   # misma hora en el cuadro actual
        elif self.prod_fit is None:
            self.prod_fit = (self.t0, 1.0 / SR)

    def set_source(self, src):
        """Cambia lo que suena sin cerrar las salidas: el retraso Bluetooth de cada parlante se mantiene
        estable (reabrirlas lo reinicia y tarda ~40 s en asentarse)."""
        old = self.source
        self.paused = False
        if getattr(old, "realtime", False) or getattr(src, "realtime", False):
            # El ritmo sigue sin saltos: la hora de producción continúa donde iba (el control por retraso
            # real la usa de referencia; un salto ahí sería un salto de sincronía).
            now_ref = self.frame_to_wall(self.produced) if self.prod_fit is not None else time.monotonic()
            self.t0 = now_ref - self.produced / SR
            self.prod_hist.clear()
        self.source = src
        self.apply_spatial(src)
        if old is not src and hasattr(old, "close"):
            try:
                old.close()
            except Exception:
                pass

    @property
    def healthy(self) -> bool:
        try:
            return self.running and all(s.stream is not None and s.stream.active for s in self.speakers)
        except Exception:          # conexión inválida (parlante que desapareció): no está sano
            return False

    def wait_settled(self, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not all(s.settled for s in self.speakers):
            time.sleep(0.02)
        time.sleep(0.1)

    def stop(self):
        global _ACTIVE
        if getattr(self, "_counted", False):
            _ACTIVE -= 1
            self._counted = False
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        for s in self.speakers:
            if s.stream:
                # abort (no stop): no espera a vaciar buffers de un parlante que quizá ya no existe
                try:
                    s.stream.abort()
                except Exception:
                    pass
                try:
                    s.stream.close()
                except Exception:
                    pass
        if hasattr(self.source, "close"):
            self.source.close()

    def set_layout(self, speakers_cfg: list, listener: dict):
        self.matrix = self._matrix(speakers_cfg, listener)
        self.apply_spatial(self.source)

    def inject(self, idx: int, signal: np.ndarray, at_frame: int):
        with self.inj_lock:
            self.injections.append((idx, int(at_frame), signal.astype(np.float32)))

    def spectrum(self, bands: int = 24) -> list:
        """Espectro de lo que suena (para el visualizador): niveles 0–1 en bandas logarítmicas."""
        n = 2048
        seg = self.ref_segment(max(0, self.produced - n), self.produced)
        if seg is None or len(seg) < n:
            return [0.0] * bands
        x = seg.sum(axis=1) * np.hanning(n)
        X = np.abs(np.fft.rfft(x)) / n
        f = np.fft.rfftfreq(n, 1 / SR)
        edges = np.geomspace(40, 16000, bands + 1)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (f >= a) & (f < b)
            v = float(X[m].max()) if m.any() else 0.0
            out.append(round(max(0.0, min(1.0, (20 * np.log10(v + 1e-9) + 70) / 60)), 3))
        return out

    def ref_segment(self, f0: int, f1: int):
        """Lo enviado a cada parlante entre los cuadros f0 y f1 (None si ya no está en memoria)."""
        if f0 < max(0, self.produced - self.ref_cap) or f1 > self.produced or f1 <= f0:
            return None
        idx = np.arange(f0, f1) % self.ref_cap
        return self.ref[idx]

    def frame_to_wall(self, frame: int) -> float:
        """Hora en que se produjo (o se producirá) un cuadro: la misma referencia que usa el control."""
        if self.prod_fit is not None:
            a, b = self.prod_fit
            return a + b * frame
        return self.t0 + frame / SR

    # --- productor -----------------------------------------------------------
    def _produce(self):
        while self.running:
            if getattr(self.source, "realtime", False):
                blk = self.source.read_available()
                if blk is None:
                    continue
            else:
                due = int((time.monotonic() - self.t0) * SR) + BLOCK - self.produced
                if due < BLOCK:
                    time.sleep(0.003)
                    continue
                src = self.source
                blk = np.zeros((BLOCK, 6), np.float32) if self.paused else src.read(BLOCK)
            self._push(blk)
            if not getattr(self.source, "realtime", False) and getattr(src, "done", False) and src is self.source:
                # Terminó el archivo: se sigue enviando silencio (los parlantes no se desconectan).
                from .sources import SilenceSource
                self.set_source(SilenceSource())
                self.ended = True

    def _push(self, blk: np.ndarray):
        n = len(blk)
        self.in_peak = max(self.in_peak * 0.9, float(np.abs(blk).max()) if n else 0.0)
        sp = self.sound
        x = sp.channels(blk)
        if self.center_extra_db:
            x[:, 2] *= 10 ** (self.center_extra_db / 20)
        k = self.master * self.headroom * 10 ** (self.extra_db / 20)
        mix = sp.speakers((x @ self.matrix) * k, x[:, render.CHANNELS.index("LFE")] * k)
        g0 = self.produced
        with self.inj_lock:
            keep = []
            for idx, start, sig in self.injections:
                a = max(start, g0)
                b = min(start + len(sig), g0 + n)
                if a < b:
                    mix[a - g0:b - g0, idx] += sig[a - start:b - start]
                if start + len(sig) > g0 + n:
                    keep.append((idx, start, sig))
            self.injections = keep
        objs = getattr(self.source, "objects", None)
        if objs:
            lst = self.config["listener"]
            az = [render.speaker_azimuth(c.cfg, lst) for c in self.speakers]
            el = [spatial3d.speaker_elevation(c.cfg, lst) for c in self.speakers]
            for ob in objs:
                if getattr(ob, "renderer", None) is None or ob.renderer.n != len(self.speakers):
                    ob.renderer = spatial3d.ObjectRenderer(len(self.speakers))
                sig = ob.block(n)
                mix += ob.renderer(sig, ob.az, getattr(ob, "el", 0.0), ob.dist, az, el) * self.master
        m, h = self.masker, self.hum
        if m is not None:
            mix += m.block(n, len(self.speakers))
        if h is not None:
            h.add(mix, g0, self.t0)
        i0 = g0 % self.ref_cap
        first = min(n, self.ref_cap - i0)
        self.ref[i0:i0 + first] = mix[:first]
        if first < n:
            self.ref[:n - first] = mix[first:]
        for i, s in enumerate(self.speakers):
            # Medidor = nivel antes de la compensación propia del parlante, o sea, lo que debería
            # llegar al punto de escucha si la calibración los igualó (no la señal enviada).
            self.peak[i] = max(self.peak[i] * 0.9, float(np.abs(mix[:, i]).max()) if n else 0.0)
            col = mix[:, i] * s.gain
            # FIR por solapamiento-suma: la cola de este bloque se suma al inicio del siguiente.
            h = s.fir
            y = oaconvolve(col, h)
            if len(s.fir_tail) != len(h) - 1:
                s.fir_tail = np.zeros(len(h) - 1)
            y[:len(h) - 1] += s.fir_tail
            s.fir_tail = y[n:].copy()
            col = y[:n]
            e = s.eq
            if e is not None:
                col, zi = sosfilt(e[0], col, zi=e[1])
                if s.eq is e:
                    s.eq = (e[0], zi)
            if s.protect is not None:
                col = s.protect(col[:, None])[:, 0]
            col = s.bass_lim(col)
            if s.bass_lim.active > 0:
                s.bass_lim_blocks += 1
            s.blocks += 1
            if np.abs(col).max(initial=0.0) > 0.75:
                s.limit_blocks += 1
            col = soft_limit(col)
            s.fifo.write(col.astype(np.float32))
        self.produced += n
        self._track_production(g0)

    # --- estado ----------------------------------------------------------------
    def status(self) -> list:
        out = []
        for i, s in enumerate(self.speakers):
            out.append({
                "name": s.name,
                "fill_err_ms": (s.lat_err * 1000) if s.lat_set is not None else (s.ema - s.target) / SR * 1000,
                "drift_ppm": (s.ratio / s.base_ratio - 1) * 1e6,
                "delay_ms": s.delay / SR * 1000,
                "gain_db": 20 * np.log10(max(s.gain, 1e-9)),
                "underruns": s.fifo.underruns,
                "resyncs": s.resyncs + s.jumps,
                "limiting_pct": round(100 * s.limit_blocks / max(1, s.blocks), 1),
                "bass_limiting_pct": round(100 * s.bass_lim_blocks / max(1, s.blocks), 1),
                "protect_hz": s.protect_hz,
                "peak": float(self.peak[i]),
            })
        return out
