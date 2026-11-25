"""CLI de Surround BT.

  python3 -m sbt devices                      lista salidas/entradas de audio
  python3 -m sbt init "JBL" "Sony" ...        crea config.json con esos parlantes (texto del nombre)
  python3 -m sbt layout                       muestra cómo se reparten los canales según posiciones
  python3 -m sbt calibrate                    mide retrasos/volúmenes con el micrófono y los guarda
  python3 -m sbt test                         ruido que recorre L, C, R, Rs, Ls
  python3 -m sbt play pelicula.mkv            reproduce un archivo (5.1 real, o estéreo->surround)
  python3 -m sbt live                         audio del sistema vía BlackHole
  python3 -m sbt ui                           interfaz en el navegador (http://localhost:8890)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sounddevice as sd

from . import calibrate as cal
from . import render
from .engine import Engine
from .sources import SR, ChannelWalkSource, FileSource, LiveInputSource, SilenceSource

from .paths import CONFIG

# Posiciones por defecto (metros). Oyente al centro de un cuarto 4x5, pantalla en y=5.
DEFAULT_SPOTS = [
    {"x": 0.8, "y": 4.6},   # frente izq
    {"x": 3.2, "y": 4.6},   # frente der
    {"x": 0.6, "y": 1.2},   # trasero izq
    {"x": 3.4, "y": 1.2},   # trasero der
    {"x": 2.0, "y": 4.7},   # centro
    {"x": 2.0, "y": 0.4},   # trasero centro
]


def load_config() -> dict:
    if not CONFIG.exists():
        sys.exit("No hay config.json. Primero: python3 -m sbt init \"Parlante1\" \"Parlante2\"")
    return json.loads(CONFIG.read_text())


def save_config(cfg: dict):
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def cmd_devices(a):
    for i, d in enumerate(sd.query_devices()):
        kind = []
        if d["max_output_channels"]:
            kind.append(f"salida {d['max_output_channels']}ch")
        if d["max_input_channels"]:
            kind.append(f"entrada {d['max_input_channels']}ch")
        print(f"[{i:2}] {d['name']}  ({', '.join(kind)})")


def cmd_init(a):
    speakers = []
    for i, dev in enumerate(a.devices):
        spot = DEFAULT_SPOTS[i % len(DEFAULT_SPOTS)]
        speakers.append({"name": dev, "device": dev, **spot, "delay_ms": 0.0, "gain_db": 0.0})
    cfg = {"room": {"width": 4.0, "depth": 5.0},
           "listener": {"x": 2.0, "y": 2.4},
           "mic": None,
           "speakers": speakers}
    save_config(cfg)
    print(f"Guardado {CONFIG}. Edita x/y de cada parlante (metros) y corre `layout`.")
    print(render.describe(speakers, cfg["listener"]))


def cmd_layout(a):
    cfg = load_config()
    print(render.describe(cfg["speakers"], cfg["listener"]))


def print_status(eng: Engine, extra=""):
    parts = []
    for s in eng.status():
        bar = "█" * int(min(s["peak"], 1) * 10)
        parts.append(f"{s['name'][:10]}: {bar:<10} desfase {s['fill_err_ms']:+5.1f}ms "
                     f"deriva {s['drift_ppm']:+6.0f}ppm")
    sys.stdout.write("\r" + " | ".join(parts) + f" {extra}   ")
    sys.stdout.flush()


def run(eng: Engine, label=None):
    eng.start()
    try:
        while eng.running:
            time.sleep(0.25)
            print_status(eng, label() if label else "")
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        print()
        for s in eng.status():
            print(f"{s['name']}: cortes {s['underruns']}, resincronizaciones {s['resyncs']}")


def cmd_calibrate(a):
    cfg = cal.run(load_config(), passes=a.passes, rounds=a.rounds)
    save_config(cfg)
    print("Calibración guardada.")


def cmd_test(a):
    cfg = load_config()
    src = ChannelWalkSource()
    current = {"ch": ""}
    src.on_change = lambda ch: current.update(ch=ch)
    run(Engine(cfg, src, a.volume), lambda: f"→ {current['ch']}")


def cmd_play(a):
    cfg = load_config()
    src = FileSource(a.file, upmix_stereo=not a.no_upmix, start=a.start)
    print(f"Archivo con {src.channels} canales"
          + (" (convirtiendo estéreo a surround)" if src.channels <= 2 and not a.no_upmix else ""))
    run(Engine(cfg, src, a.volume))


def cmd_live(a):
    cfg = load_config()
    inp = a.input.lower()
    loop = [sp for sp in cfg["speakers"] if inp in str(sp["device"]).lower()]
    if loop:
        print("Aviso: " + ", ".join(sp["name"] for sp in loop) + " usa el mismo dispositivo que la entrada; "
              "se omite (CoreAudio no permite leer y escribir BlackHole a la vez aquí).")
        cfg["speakers"] = [sp for sp in cfg["speakers"] if sp not in loop]
        if not cfg["speakers"]:
            sys.exit("No quedan parlantes de salida.")
    src = LiveInputSource(a.input, upmix_stereo=not a.no_upmix)
    print(f"Escuchando '{sd.query_devices(src.stream.device)['name']}'. Pon la salida de la Mac en "
          f"BlackHole (o en un dispositivo de salida múltiple que lo incluya). Ctrl+C para salir.")
    run(Engine(cfg, src, a.volume))


def cmd_ui(a):
    from .server import serve
    serve("127.0.0.1" if a.local else "0.0.0.0", a.port)


def main():
    p = argparse.ArgumentParser(prog="sbt", description="Surround virtual con varios parlantes Bluetooth")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices").set_defaults(fn=cmd_devices)
    s = sub.add_parser("init")
    s.add_argument("devices", nargs="+")
    s.set_defaults(fn=cmd_init)
    sub.add_parser("layout").set_defaults(fn=cmd_layout)
    s = sub.add_parser("ui", help="interfaz en el navegador")
    s.add_argument("--port", type=int, default=8890)
    s.add_argument("--local", action="store_true", help="solo desde esta Mac (sin control desde el iPhone)")
    s.set_defaults(fn=cmd_ui)
    s = sub.add_parser("calibrate")
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--rounds", type=int, default=2)
    s.set_defaults(fn=cmd_calibrate)
    for name, fn in (("test", cmd_test), ("play", cmd_play), ("live", cmd_live)):
        s = sub.add_parser(name)
        s.add_argument("--volume", type=float, default=0.0, help="dB")
        if name == "play":
            s.add_argument("file")
            s.add_argument("--start", type=float, default=0.0)
        if name == "live":
            s.add_argument("--input", default="BlackHole")
        if name in ("play", "live"):
            s.add_argument("--no-upmix", action="store_true")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
