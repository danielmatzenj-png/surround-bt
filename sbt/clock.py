"""Horas de CoreAudio → reloj de pared (time.monotonic).

PortAudio informa en cada callback cuándo se capturó (entrada) o cuándo va a sonar (salida) la primera
muestra del bloque, en el reloj del sistema (mach_absolute_time, en segundos). Esa hora es exacta (±10 µs);
la hora a la que llega el callback no: macOS los entrega a tirones (±3–7 ms cuando convierte frecuencia)."""
from __future__ import annotations

import ctypes
import time


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_lib.mach_absolute_time.restype = ctypes.c_uint64
_tb = _Timebase()
_lib.mach_timebase_info(ctypes.byref(_tb))
_K = _tb.numer / _tb.denom * 1e-9


def host_now() -> float:
    """Ahora, en el reloj de CoreAudio (s)."""
    return _lib.mach_absolute_time() * _K


def to_mono(host_t: float) -> float:
    """Una hora de CoreAudio pasada a time.monotonic (la diferencia entre ambos se lee en el momento)."""
    return host_t + (time.monotonic() - host_now())


def adc_time(t) -> float | None:
    """Hora (monotonic) de captura de la primera muestra del bloque de entrada; None si no la informa."""
    adc = getattr(t, "inputBufferAdcTime", 0.0)
    return to_mono(adc) if adc else None


def dac_time(t) -> float | None:
    """Hora (monotonic) en que sonará la primera muestra del bloque de salida; None si no la informa."""
    dac = getattr(t, "outputBufferDacTime", 0.0)
    return to_mono(dac) if dac else None
