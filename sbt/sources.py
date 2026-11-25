"""Fuentes de audio. Todas entregan bloques float32 (n, 6) en orden L R C LFE Ls Rs a SR."""
from __future__ import annotations

import queue
import re
import subprocess

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from .sources_base import SR


class Upmixer:
    """Estéreo -> 5.1: frente intacto, centro suave, ambiente (L-R) retrasado a los traseros, LFE por pasabajos."""

    def __init__(self, sr: int = SR, center: float = 0.35, surround: float = 0.6):
        self.center, self.surround = center, surround
        self.b_lfe, self.a_lfe = butter(2, 120 / (sr / 2))
        self.b_sur, self.a_sur = butter(1, 7000 / (sr / 2))
        self.z_lfe = lfilter_zi(self.b_lfe, self.a_lfe) * 0
        self.z_sl = lfilter_zi(self.b_sur, self.a_sur) * 0
        self.z_sr = lfilter_zi(self.b_sur, self.a_sur) * 0
        self.dl = int(0.012 * sr)
        self.dr = int(0.017 * sr)  # retrasos distintos = más sensación de espacio
        self.hist = np.zeros(max(self.dl, self.dr), dtype=np.float32)

    def __call__(self, st: np.ndarray) -> np.ndarray:
        L, R = st[:, 0], st[:, 1]
        mid = 0.5 * (L + R)
        side = 0.5 * (L - R)
        ext = np.concatenate([self.hist, side])
        n = len(side)
        h = len(self.hist)
        sl_raw = ext[h - self.dl:h - self.dl + n]
        sr_raw = -ext[h - self.dr:h - self.dr + n]
        self.hist = ext[-h:]
        sl, self.z_sl = lfilter(self.b_sur, self.a_sur, sl_raw, zi=self.z_sl)
        sr_, self.z_sr = lfilter(self.b_sur, self.a_sur, sr_raw, zi=self.z_sr)
        lfe, self.z_lfe = lfilter(self.b_lfe, self.a_lfe, mid, zi=self.z_lfe)
        out = np.empty((n, 6), dtype=np.float32)
        out[:, 0] = L
        out[:, 1] = R
        out[:, 2] = mid * self.center
        out[:, 3] = lfe
        out[:, 4] = sl * self.surround
        out[:, 5] = sr_ * self.surround
        return out


def make_upmixer():
    from .upmix import PAUpmixer
    return PAUpmixer()


def to_51(block: np.ndarray, upmix) -> np.ndarray:
    ch = block.shape[1]
    if ch == 6:
        return block.astype(np.float32)
    if ch == 8:  # 7.1: FL FR FC LFE BL BR SL SR -> suma laterales y traseros
        out = block[:, :6].astype(np.float32).copy()
        out[:, 4] = 0.707 * (block[:, 4] + block[:, 6])
        out[:, 5] = 0.707 * (block[:, 5] + block[:, 7])
        return out
    if ch == 1:
        block = np.repeat(block, 2, axis=1)
    return upmix(block[:, :2]) if upmix else np.pad(block[:, :2], ((0, 0), (0, 4)))


class SilenceSource:
    realtime = False

    def read(self, n):
        return np.zeros((n, 6), dtype=np.float32)


class ChannelWalkSource:
    """Ruido rosa que recorre los canales L -> C -> R -> Rs -> Ls para escuchar la colocación."""
    realtime = False
    order = [0, 2, 1, 5, 4]
    names = ["L", "C", "R", "Rs", "Ls"]

    def __init__(self, seconds_each: float = 1.5, loops: int | None = None):
        self.len = int(seconds_each * SR)
        self.pos = 0
        self.total = None if loops is None else loops * len(self.order) * self.len
        self.done = False
        rng = np.random.default_rng(1)
        white = rng.standard_normal(SR * 2).astype(np.float32)
        spec = np.fft.rfft(white)
        f = np.arange(len(spec))
        spec[1:] /= np.sqrt(f[1:])
        pink = np.fft.irfft(spec).astype(np.float32)
        self.noise = 0.25 * pink / np.abs(pink).max()
        self.on_change = None

    def read(self, n):
        out = np.zeros((n, 6), dtype=np.float32)
        idx = (self.pos + np.arange(n))
        slot = (idx // self.len) % len(self.order)
        within = idx % self.len
        env = np.clip(np.minimum(within, self.len - within) / 2400.0, 0, 1) * (within < self.len * 0.8)
        sig = self.noise[idx % len(self.noise)] * env
        for s in np.unique(slot):
            mask = slot == s
            out[mask, self.order[s]] = sig[mask]
            if self.on_change and within[mask][0] == 0:
                self.on_change(self.names[s])
        self.pos += n
        if self.total is not None and self.pos >= self.total:
            self.done = True
        return out


def probe_duration(path: str, ffmpeg: str) -> float:
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", p.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def probe_channels(path: str, ffmpeg: str) -> int:
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Audio: [^\n]*?, \d+ Hz, ([^,\n]+)", p.stderr)
    if not m:
        raise RuntimeError(f"No encontré pista de audio en {path}")
    layout = m.group(1).strip()
    if layout.startswith("7.1"):
        return 8
    if layout.startswith("5.1"):
        return 6
    if layout.startswith("mono"):
        return 1
    mc = re.match(r"(\d+) channels", layout)
    if mc:
        return int(mc.group(1))
    return 2


class FileSource:
    """Decodifica cualquier archivo (MKV/MP4/MP3/FLAC...) con ffmpeg estático."""
    realtime = False

    def __init__(self, path: str, upmix_stereo: bool = True, start: float = 0.0):
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        self.channels = probe_channels(path, ff)
        ch = self.channels if self.channels in (1, 2, 6, 8) else 2
        self.ch = ch
        self.proc = subprocess.Popen(
            [ff, "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", path, "-vn",
             "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"],
            stdout=subprocess.PIPE)
        self.upmix = make_upmixer() if upmix_stereo else None
        self.done = False
        self.path, self.start = path, float(start)
        self.duration = probe_duration(path, ff)
        self.frames = 0            # muestras realmente leídas del archivo (para la posición)

    @property
    def position(self) -> float:
        return self.start + self.frames / SR

    def read(self, n):
        need = n * self.ch * 4
        data = b""
        while len(data) < need:
            chunk = self.proc.stdout.read(need - len(data))
            if not chunk:
                self.done = True
                break
            data += chunk
        arr = np.frombuffer(data[:len(data) // (self.ch * 4) * self.ch * 4], dtype=np.float32)
        arr = arr.reshape(-1, self.ch)
        self.frames += len(arr)
        if len(arr) < n:
            arr = np.pad(arr, ((0, n - len(arr)), (0, 0)))
        return to_51(arr, self.upmix)

    def close(self):
        self.proc.kill()


class StreamResampler:
    """Remuestreo continuo multicanal (interpolación lineal) para entradas que no corren a SR."""

    def __init__(self, in_sr: float, out_sr: float = SR):
        self.step = in_sr / out_sr
        self.carry = None
        self.pos = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if abs(self.step - 1.0) < 1e-9:
            return x
        buf = x if self.carry is None else np.concatenate([self.carry, x])
        n_out = int(np.floor((len(buf) - 1 - self.pos) / self.step)) + 1
        if n_out <= 0:
            self.carry = buf
            return np.zeros((0, x.shape[1]), dtype=np.float32)
        p = self.pos + self.step * np.arange(n_out)
        i0 = np.floor(p).astype(np.int64)
        t = (p - i0)[:, None].astype(np.float32)
        i1 = np.minimum(i0 + 1, len(buf) - 1)
        out = buf[i0] * (1 - t) + buf[i1] * t
        nxt = self.pos + self.step * n_out
        keep = int(np.floor(nxt))
        self.carry = buf[keep:]
        self.pos = nxt - keep
        return out.astype(np.float32)


class LiveInputSource:
    """Captura el audio del sistema desde BlackHole (o cualquier entrada). El reloj de la entrada manda."""
    realtime = True

    def __init__(self, device, upmix_stereo: bool = True):
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        self.ch = min(int(info["max_input_channels"]), 8)
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self.upmix = make_upmixer() if upmix_stereo else None
        # CoreAudio entrega a la frecuencia real del dispositivo (BlackHole suele estar a 44.1 kHz).
        self.native = float(info["default_samplerate"])
        self.resample = StreamResampler(self.native)
        self.stream = sd.InputStream(device=device, channels=self.ch, samplerate=self.native, latency="low",
                                     dtype="float32", blocksize=512, callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def read_available(self, timeout=0.1):
        try:
            blk = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        # Si se acumuló audio (p. ej. al arrancar), se descarta lo viejo: cada bloque guardado es retraso.
        while self.q.qsize() > 2:
            try:
                blk = self.q.get_nowait()
            except queue.Empty:
                break
        blk = self.resample(blk)
        if not len(blk):
            return None
        return to_51(blk, self.upmix)

    def close(self):
        self.stream.stop()
        self.stream.close()


class SoundObject:
    """Un sonido ubicado en un punto del cuarto (azimut y distancia respecto a ti)."""

    def __init__(self, kind: str = "pink"):
        self.kind = kind
        self.az = 0.0
        self.el = 0.0              # elevación (grados): + arriba, − abajo
        self.dist = 1.5
        self.renderer = None
        self.orbit = True
        self.speed = 45.0          # grados por segundo en órbita
        self.pos = 0
        rng = np.random.default_rng(3)
        white = rng.standard_normal(SR * 2).astype(np.float32)
        spec = np.fft.rfft(white)
        spec[1:] /= np.sqrt(np.arange(1, len(spec)))
        pink = np.fft.irfft(spec).astype(np.float32)
        self.pink = 0.3 * pink / np.abs(pink).max()
        t = np.arange(int(0.9 * SR)) / SR
        bell = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d)
                   for f, a, d in ((660, 0.5, 4), (1320, 0.25, 6), (1980, 0.12, 9), (2640, 0.06, 12)))
        self.bell = (0.4 * bell).astype(np.float32)

    def block(self, n: int) -> np.ndarray:
        idx = self.pos + np.arange(n)
        if self.kind == "bell":
            sig = self.bell[idx % len(self.bell)] * ((idx % SR) < len(self.bell))
        else:   # ruido rosa en pulsos: se ubica mejor que un ruido continuo
            env = np.clip(np.sin(np.pi * (idx % (SR // 2)) / (SR // 2)) * 1.6, 0, 1)
            sig = self.pink[idx % len(self.pink)] * env
        if self.orbit:
            self.az = (self.az + self.speed * n / SR + 180) % 360 - 180
        self.pos += n
        return sig.astype(np.float32) * 1.5          # la distancia la aplica la esfera (spatial3d)


class ObjectSource:
    """Modo 3D: los canales van en silencio y el sonido-objeto se ubica directo entre parlantes."""
    realtime = False

    def __init__(self, kind="pink"):
        self.objects = [SoundObject(kind)]

    def read(self, n):
        return np.zeros((n, 6), dtype=np.float32)
