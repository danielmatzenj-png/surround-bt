"""Volumen de hardware de cada salida (CoreAudio vía ctypes). En parlantes Bluetooth este volumen
es el del propio parlante: macOS lo guarda por dispositivo y se lo envía al conectarse."""
from __future__ import annotations

import ctypes
import ctypes.util

_ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))
_cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


class _Addr(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


SYSTEM = 1
GLOBAL, OUTPUT = _fourcc("glob"), _fourcc("outp")
DEVICES, NAME, VOLUME = _fourcc("dev#"), _fourcc("lnam"), _fourcc("volm")
STREAMS = _fourcc("stm#")

_ca.AudioObjectGetPropertyDataSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                               ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_ca.AudioObjectGetPropertyData.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                           ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
_ca.AudioObjectSetPropertyData.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                           ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
_ca.AudioObjectHasProperty.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr)]
_ca.AudioObjectHasProperty.restype = ctypes.c_bool
_cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
_cf.CFRelease.argtypes = [ctypes.c_void_p]


def _devices() -> list:
    addr = _Addr(DEVICES, GLOBAL, 0)
    size = ctypes.c_uint32(0)
    _ca.AudioObjectGetPropertyDataSize(SYSTEM, ctypes.byref(addr), 0, None, ctypes.byref(size))
    n = size.value // 4
    ids = (ctypes.c_uint32 * n)()
    _ca.AudioObjectGetPropertyData(SYSTEM, ctypes.byref(addr), 0, None, ctypes.byref(size), ids)
    return list(ids)


def _name(dev: int) -> str:
    addr = _Addr(NAME, GLOBAL, 0)
    ref = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ref))
    if _ca.AudioObjectGetPropertyData(dev, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(ref)):
        return ""
    buf = ctypes.create_string_buffer(256)
    _cf.CFStringGetCString(ref, buf, 256, 0x08000100)  # UTF-8
    _cf.CFRelease(ref)
    return buf.value.decode("utf-8", "replace")


def _has_output(dev: int) -> bool:
    addr = _Addr(STREAMS, OUTPUT, 0)
    size = ctypes.c_uint32(0)
    _ca.AudioObjectGetPropertyDataSize(dev, ctypes.byref(addr), 0, None, ctypes.byref(size))
    return size.value > 0


def _find(name: str):
    for d in _devices():
        if _name(d) == name and _has_output(d):
            return d
    return None


def _elements(dev: int) -> list:
    """Elemento 0 (volumen maestro) si existe; si no, los canales 1 y 2."""
    for elems in ([0], [1, 2]):
        if all(_ca.AudioObjectHasProperty(dev, ctypes.byref(_Addr(VOLUME, OUTPUT, e))) for e in elems):
            return elems
    return []


def get_volume(name: str):
    """0.0–1.0, o None si el dispositivo no expone volumen."""
    dev = _find(name)
    if dev is None:
        return None
    vals = []
    for e in _elements(dev):
        v = ctypes.c_float()
        size = ctypes.c_uint32(4)
        if not _ca.AudioObjectGetPropertyData(dev, ctypes.byref(_Addr(VOLUME, OUTPUT, e)), 0, None,
                                              ctypes.byref(size), ctypes.byref(v)):
            vals.append(v.value)
    return sum(vals) / len(vals) if vals else None


def set_volume(name: str, value: float) -> bool:
    dev = _find(name)
    if dev is None:
        return False
    ok = False
    for e in _elements(dev):
        v = ctypes.c_float(max(0.0, min(1.0, value)))
        ok |= _ca.AudioObjectSetPropertyData(dev, ctypes.byref(_Addr(VOLUME, OUTPUT, e)), 0, None, 4,
                                             ctypes.byref(v)) == 0
    return ok


def default_output_name() -> str:
    """Salida actual de la Mac, leída directo de CoreAudio (sin reiniciar PortAudio)."""
    addr = _Addr(_fourcc("dOut"), GLOBAL, 0)
    dev = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    _ca.AudioObjectGetPropertyData(SYSTEM, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(dev))
    return _name(dev.value)


INPUT = _fourcc("inpt")


def _has_input(dev: int) -> bool:
    addr = _Addr(STREAMS, INPUT, 0)
    size = ctypes.c_uint32(0)
    _ca.AudioObjectGetPropertyDataSize(dev, ctypes.byref(addr), 0, None, ctypes.byref(size))
    return size.value > 0


def default_input_name() -> str:
    addr = _Addr(_fourcc("dIn "), GLOBAL, 0)
    dev = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    _ca.AudioObjectGetPropertyData(SYSTEM, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(dev))
    return _name(dev.value)


def use_builtin_mic(speaker_names) -> str | None:
    """Si la entrada de la Mac es el micrófono de un parlante Bluetooth, la cambia al de la Mac:
    usar ese micrófono pasa el parlante a «modo llamada» (16 kHz) y deja de sonar bien."""
    cur = default_input_name()
    if not any(n and n.lower() in cur.lower() for n in speaker_names):
        return None
    want = preferred_mic()
    for d in _devices():
        name = _name(d)
        if _has_input(d) and name == want:
            dev = ctypes.c_uint32(d)
            _ca.AudioObjectSetPropertyData(SYSTEM, ctypes.byref(_Addr(_fourcc("dIn "), GLOBAL, 0)), 0, None, 4,
                                           ctypes.byref(dev))
            return f"{cur} → {name}"
    return None


def output_ids(names) -> dict:
    """Identificador actual de CoreAudio de cada salida (cambia si el parlante se reconecta)."""
    out = {}
    devs = [(d, _name(d)) for d in _devices()]
    for n in names:
        out[n] = next((d for d, nm in devs if nm == n and _has_output(d)), None)
    return out


TRANSPORT = _fourcc("tran")
_KINDS = {_fourcc("blue"): "bluetooth", _fourcc("blea"): "bluetooth", _fourcc("bltn"): "builtin",
          _fourcc("virt"): "virtual", _fourcc("usb "): "usb", _fourcc("airp"): "airplay",
          _fourcc("hdmi"): "hdmi", _fourcc("dprt"): "displayport", _fourcc("aggr"): "aggregate"}


def _transport(dev: int) -> str:
    v = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    if _ca.AudioObjectGetPropertyData(dev, ctypes.byref(_Addr(TRANSPORT, GLOBAL, 0)), 0, None,
                                      ctypes.byref(size), ctypes.byref(v)):
        return "unknown"
    return _KINDS.get(v.value, "unknown")


def list_devices() -> list:
    """Todos los dispositivos con su tipo de conexión (para el asistente de primer uso)."""
    out = []
    for d in _devices():
        out.append({"name": _name(d), "kind": _transport(d), "output": _has_output(d), "input": _has_input(d)})
    return out


def preferred_mic() -> str | None:
    """Micrófono para calibrar: el integrado si hay; si no (Mac mini), uno USB; nunca uno Bluetooth
    (lo pasaría a modo llamada)."""
    devs = [d for d in list_devices() if d["input"]]
    for kind in ("builtin", "usb", "unknown"):
        for d in devs:
            if d["kind"] == kind:
                return d["name"]
    return None


def loopback_name() -> str | None:
    """Dispositivo virtual para capturar el audio de la Mac (BlackHole, Loopback, Soundflower…)."""
    for d in list_devices():
        if d["kind"] == "virtual" and d["input"] and d["output"]:
            return d["name"]
    return None


def set_default_output(name: str) -> bool:
    for d in _devices():
        if _name(d) == name and _has_output(d):
            dev = ctypes.c_uint32(d)
            return _ca.AudioObjectSetPropertyData(SYSTEM, ctypes.byref(_Addr(_fourcc("dOut"), GLOBAL, 0)), 0,
                                                  None, 4, ctypes.byref(dev)) == 0
    return False
