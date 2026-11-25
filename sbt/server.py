"""Servidor local de la interfaz: sirve ui/index.html y una API JSON que controla el motor."""
from __future__ import annotations

import json
import secrets
import socket
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import calibrate as cal
from . import coreaudio
from . import library, placement
from . import dsp
from . import eq as eqmod
from . import render
from .engine import Engine
from .sources import ChannelWalkSource, FileSource, LiveInputSource, ObjectSource, SilenceSource

from .paths import BIN, CONFIG, PROFILES, RES as ROOT, UI
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".webm", ".mkv"}
MEDIA_EXT = {".mkv", ".mp4", ".mov", ".m4v", ".avi", ".mp3", ".m4a", ".flac", ".wav", ".aac", ".ogg", ".ac3"}


def default_config() -> dict:
    return {"room": {"width": 4.0, "depth": 5.0}, "listener": {"x": 2.0, "y": 2.4},
            "mic": None, "speakers": [], "eq": {"preset": "flat", "bands": eqmod.PRESETS["flat"]},
            "eq_match": True}


class Controller:
    def __init__(self):
        self.lock = threading.RLock()
        self.cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else default_config()
        # BlackHole (u otro virtual) como parlante: el audio de la Mac entra por ahí, así que mandarle sonido
        # arma un circuito que traba todo. Si quedó de una versión vieja, se quita.
        self.cfg["speakers"] = [sp for sp in self.cfg.get("speakers", []) if not is_virtual(sp.get("device", ""))]
        self.engine = None
        if dsp.sound_cfg(self.cfg).get("pair_volume", True):
            self.even_pairs()
        self.cfg.setdefault("eq", {"preset": "flat", "bands": list(eqmod.PRESETS["flat"])})
        self.cfg.setdefault("eq_match", True)
        if not self.cfg.get("remote_token"):
            self.cfg["remote_token"] = secrets.token_urlsafe(9)
            self.save()
        self.engine: Engine | None = None
        self.mode = None            # "test" | "file" | "live"
        self.file = None
        self.volume_db = -6.0
        self.channel = ""
        self.error = ""
        self.cal = {"running": False, "log": [], "done_at": None, "waiting": None}
        self.cal_continue = threading.Event()
        self.monitor = None
        self.obj_kind = "pink"
        self.engine_sig = None
        self.engine_ids = None
        self.missing = []
        self.engine_error = ""
        self.engine_since = 0.0
        self.last_open_try = 0.0

    def save(self):
        CONFIG.write_text(json.dumps(self.cfg, indent=2, ensure_ascii=False))

    # --- reproducción ----------------------------------------------------------
    # --- motor persistente ------------------------------------------------------------
    # Las salidas Bluetooth quedan abiertas mientras la app está abierta (enviando silencio si no suena
    # nada): reabrirlas reinicia el retraso de cada parlante, que tarda ~40 s en asentarse (medido).
    def _sig(self):
        return tuple((s["name"], s["device"]) for s in self.cfg["speakers"])

    def ensure_engine(self):
        with self.lock:
            if self.engine is not None and self.engine_sig == self._sig() and self.engine.healthy:
                return self.engine
            self.missing = []
            self.shutdown()
            if not self.cfg["speakers"]:
                return None
            if time.monotonic() - self.last_open_try < 5 and self.engine_error:
                raise RuntimeError(self.engine_error)
            self.last_open_try = time.monotonic()
            cfg = json.loads(json.dumps(self.cfg))
            # Arrancar con los parlantes que estén; los que falten se suman solos al aparecer.
            try:
                ids = coreaudio.output_ids([sp["device"] for sp in cfg["speakers"]])
                present = [sp for sp in cfg["speakers"] if ids.get(sp["device"]) is not None]
            except Exception:
                present = cfg["speakers"]
            self.missing = [sp["name"] for sp in cfg["speakers"] if sp not in present]
            if not present:
                self.engine_error = "Ningún parlante está conectado. Enciéndelos; se conectan solos."
                raise RuntimeError(self.engine_error)
            cfg["speakers"] = present
            self.fix_input()
            self.level_hardware()
            try:
                eng = Engine(cfg, SilenceSource(), self.volume_db)
                for s, sc in zip(eng.speakers, cfg["speakers"]):
                    s.mute = bool(sc.get("mute"))
                eng.start()
            except Exception as e:
                self.engine_error = str(e)
                raise
            self.engine, self.engine_sig, self.engine_error = eng, self._sig(), ""
            if self.missing:
                self.error = f"Sonando sin {', '.join(self.missing)} (no está conectado); se sumará solo al volver."
            self.engine_since = time.monotonic()
            try:
                self.engine_ids = coreaudio.output_ids([s["device"] for s in cfg["speakers"]])
            except Exception:
                self.engine_ids = None
            return eng

    def shutdown(self):
        """Cierra de verdad las salidas (al cambiar parlantes o cargar un perfil)."""
        with self.lock:
            if self.monitor is not None:
                self.monitor.stop()
                self.monitor = None
            if self.engine is not None:
                try:
                    self.engine.stop()
                finally:
                    self.engine = None
            self.mode = None

    def start(self, mode: str, file: str | None = None):
        with self.lock:
            self.stop()
            if self.cal["running"]:
                raise RuntimeError("Espera a que termine la calibración.")
            if not self.cfg["speakers"]:
                raise RuntimeError("Agrega al menos un parlante.")
            if mode == "test":
                src = ChannelWalkSource(loops=2)
                src.on_change = lambda ch: setattr(self, "channel", ch)
            elif mode == "file":
                if not file or not Path(file).expanduser().exists():
                    raise RuntimeError("Elige un archivo que exista.")
                src = FileSource(str(Path(file).expanduser()))
            elif mode == "mask":
                src = SilenceSource()
            elif mode == "3d":
                src = ObjectSource(self.obj_kind)
            elif mode == "live":
                lb = coreaudio.loopback_name()
                if not lb:
                    raise RuntimeError("Para el audio de la Mac hace falta BlackHole (gratis). Ve a Ayuda → «Audio de la Mac».")
                if any(lb.lower() in str(s["device"]).lower() for s in self.cfg["speakers"]):
                    raise RuntimeError(f"Quita {lb} de la lista de parlantes para usar el modo en vivo.")
                check_live_output(lb)
                src = LiveInputSource(lb)
            else:
                raise RuntimeError("Modo desconocido")
            try:
                eng = self.ensure_engine()
            except Exception:
                if hasattr(src, "close"):
                    src.close()
                raise
            eng.set_source(src)
            self.mode, self.file, self.error = mode, file, ""
            self.channel = ""
            self._sync_monitor()

    def _sync_monitor(self):
        """El micrófono solo se usa si hay alguna función de ruido activa y algo sonando."""
        snd = dsp.sound_cfg(self.cfg)
        want = self.engine is not None and self.mode not in (None, "mask_off") and (
            snd["adaptive"] or snd["mask"] or snd["hum"] or (snd["autosync"] and len(self.cfg["speakers"]) > 1))
        if want and self.monitor is None:
            from .monitor import RoomMonitor
            try:
                self.monitor = RoomMonitor(self.engine, self.cfg)
            except Exception as e:
                self.error = f"No pude abrir el micrófono: {e}"
        elif not want and self.monitor is not None:
            self.monitor.stop()
            self.monitor = None

    def stop(self):
        """Deja de sonar, pero mantiene las salidas abiertas (en silencio)."""
        with self.lock:
            if self.monitor is not None:
                self.monitor.stop()
                self.monitor = None
            if self.engine is not None:
                self.engine.set_source(SilenceSource())
            self.mode = None
            self.channel = ""

    def set_volume(self, db: float):
        self.volume_db = max(-60.0, min(12.0, float(db)))
        if self.engine:
            self.engine.set_master(self.volume_db)

    # --- sonido, archivo, utilidades -----------------------------------------------
    def set_sound(self, data: dict):
        with self.lock:
            snd = dsp.sound_cfg(self.cfg)
            for k, v in data.items():
                if k not in dsp.DEFAULT_SOUND:
                    continue
                if k in ("center_db", "surround_db", "lfe_db"):
                    v = round(max(-12.0, min(12.0, float(v))), 1)
                elif k == "dialogue":
                    v = max(0, min(2, int(v)))
                elif k == "spatial":
                    v = max(0, min(3, int(v)))
                elif k == "party":
                    v = bool(v)
                elif k == "crossover":
                    v = int(max(60, min(200, float(v))))
                    snd["crossover_auto"] = False      # lo eligió el usuario a mano
                elif k == "bass_mode":
                    v = "auto" if v == "auto" else "off"
                elif k == "mask_db":
                    v = round(max(-15.0, min(6.0, float(v))), 1)
                else:
                    v = bool(v)
                snd[k] = v
            self.cfg["sound"] = snd
            if data.get("pair_volume"):
                self.even_pairs()
            self.save()
            if self.engine:
                self.engine.set_sound(snd)
                if self.monitor is not None:
                    self.monitor.cfg = self.cfg
            elif snd["mask"] and data.get("mask") and not self.cal["running"] and self.cfg["speakers"]:
                self.start("mask")          # enmascarar sin nada sonando: arranca solo
                return
            if self.mode == "mask" and not snd["mask"]:
                self.stop()
                return
            self._sync_monitor()

    def pause(self, paused: bool):
        if self.engine and self.mode == "file":
            self.engine.paused = bool(paused)

    def seek(self, seconds: float):
        with self.lock:
            if not (self.engine and self.mode == "file"):
                return
            old = self.engine.source
            dur = getattr(old, "duration", 0) or 0
            pos = max(0.0, min(float(seconds), max(0.0, dur - 1)))
            self.engine.source = FileSource(old.path, start=pos)
            old.close()

    def identify(self, idx: int):
        """Dos notas solo en ese parlante, para saber cuál es cuál."""
        spk = self.cfg["speakers"][idx]
        chime = cal.chime()
        with self.lock:
            if self.engine:
                for i, s in enumerate(self.engine.speakers):
                    if s.name == spk["name"]:
                        self.engine.inject(i, chime, self.engine.produced + int(0.15 * 48000))
                return
            if self.cal["running"]:
                raise RuntimeError("Espera a que termine la calibración.")
            eng = self.ensure_engine()
            for i, s in enumerate(eng.speakers):
                if s.name == spk["name"]:
                    eng.inject(i, chime, eng.produced + int(0.15 * 48000))

    def quick_sync(self):
        with self.lock:
            if self.cal["running"]:
                return
            self.stop()
            self.cal = {"running": True, "log": ["Resincronizando tiempos…"], "done_at": None, "waiting": None}
            cal.CANCEL.clear()

        def work():
            try:
                self.level_hardware(self.cal["log"].append)
                eng = self.ensure_engine()
                cfg = cal.quick_sync(json.loads(json.dumps(self.cfg)), log=self.cal["log"].append, engine=eng)
                with self.lock:
                    for spk, new in zip(self.cfg["speakers"], cfg["speakers"]):
                        spk["delay_ms"] = new["delay_ms"]
                    if "acoustic_latency_ms" in cfg:
                        self.cfg["acoustic_latency_ms"] = cfg["acoustic_latency_ms"]
                    self.save()
            except cal.Cancelled:
                self.cal["log"].append("Cancelado; no se guardó nada.")
            except Exception as e:
                self.cal["log"].append(f"Error: {e}")
            finally:
                self.cal["running"] = False
                self.cal["done_at"] = time.time()
        threading.Thread(target=work, daemon=True).start()

    def av_latency(self) -> float:
        """Segundos desde que el audio se lee del archivo hasta que llega a tus oídos."""
        lat = float(self.cfg.get("acoustic_latency_ms") or 450.0) / 1000
        src = self.engine.source if self.engine else None
        if getattr(src, "upmix", None) is not None:
            lat += 512 / 48000                      # el upmix agrega medio cuadro
        return lat + float(self.cfg.get("av_offset_ms", 0.0)) / 1000

    def set_av_offset(self, ms: float):
        with self.lock:
            self.cfg["av_offset_ms"] = round(max(-500.0, min(500.0, float(ms))), 0)
            self.save()

    def set_scene(self, name: str):
        sc = dsp.SCENES.get(name)
        if not sc:
            raise RuntimeError("Escena desconocida")
        self.set_sound(sc["sound"])
        self.set_eq({"preset": sc["eq"]})
        with self.lock:
            self.cfg["scene"] = name
            self.save()

    def set_object(self, data: dict):
        """Modo 3D: ubicar el sonido (x, y en metros), activar la órbita o cambiar el sonido."""
        with self.lock:
            if "kind" in data:
                self.obj_kind = "bell" if data["kind"] == "bell" else "pink"
            src = self.engine.source if self.engine else None
            obs = getattr(src, "objects", None)
            if not obs:
                return
            ob = obs[0]
            if "kind" in data:
                ob.kind = self.obj_kind
            if "orbit" in data:
                ob.orbit = bool(data["orbit"])
            if "x" in data and "y" in data:
                lst = self.cfg["listener"]
                pt = {"x": float(data["x"]), "y": float(data["y"])}
                ob.az = render.speaker_azimuth(pt, lst)
                ob.dist = render.speaker_distance(pt, lst)
                ob.orbit = False
            # Esfera: posición directa en coordenadas del oyente (grados y metros).
            if "az" in data:
                ob.az = ((float(data["az"]) + 180.0) % 360.0) - 180.0
                ob.orbit = False
            if "el" in data:
                ob.el = max(-45.0, min(90.0, float(data["el"])))
            if "dist" in data:
                ob.dist = max(0.3, min(15.0, float(data["dist"])))

    # --- asistente de primer uso ----------------------------------------------------------
    LAYOUTS = {  # ángulos (grados, 0 = frente) por disposición
        "single": [0], "front2": [-30, 30], "sides": [-90, 90], "front_center": [-30, 0, 30],
        "center_sides": [-90, 0, 90], "front_back": [-30, 30, -130, 130], "five": [-30, 0, 30, -115, 115],
    }

    def setup_info(self) -> dict:
        devs = coreaudio.list_devices()
        used = {s["device"] for s in self.cfg["speakers"]}
        outs = [{"name": d["name"], "kind": d["kind"], "selected": d["name"] in used}
                for d in devs if d["output"] and d["kind"] not in ("virtual", "aggregate")
                and "salida múltiple" not in d["name"].lower() and "multi-output" not in d["name"].lower()]
        outs.sort(key=lambda d: (d["kind"] != "bluetooth", d["name"]))
        return {"outputs": outs, "loopback": coreaudio.loopback_name(), "mic": coreaudio.preferred_mic(),
                "mac_output": coreaudio.default_output_name(), "onboarded": bool(self.cfg.get("onboarded")),
                "layouts": list(self.LAYOUTS)}

    def setup_speakers(self, devices: list, layout: str):
        import math
        angles = self.LAYOUTS.get(layout)
        if not devices:
            raise RuntimeError("Elige al menos un parlante.")
        if not angles or len(angles) != len(devices):     # «lo ubico yo»: en círculo
            angles = [round(-150 + 300 * i / max(1, len(devices) - 1)) if len(devices) > 1 else 0
                      for i in range(len(devices))]
        with self.lock:
            self.shutdown()
            old = {s["device"]: s for s in self.cfg["speakers"]}
            room = {"width": 4.0, "depth": 5.0}
            lst = {"x": 2.0, "y": 2.4}
            spk = []
            for dev, a in zip(devices, angles):
                r = 1.9 if abs(a) >= 60 else 2.2
                x = min(room["width"] - 0.2, max(0.2, lst["x"] + r * math.sin(math.radians(a))))
                y = min(room["depth"] - 0.2, max(0.2, lst["y"] + r * math.cos(math.radians(a))))
                base = {k: v for k, v in old.get(dev, {}).items() if k not in ("x", "y")}
                spk.append({"name": dev, "device": dev, "delay_ms": 0.0, "gain_db": 0.0, **base,
                            "x": round(x, 2), "y": round(y, 2)})
            self.cfg.update({"room": room, "listener": lst, "speakers": spk, "layout": layout})
            self.save()
        self.warm_up()

    def identify_device(self, device: str):
        """Asistente: dos notas en un dispositivo aunque todavía no esté en la lista de parlantes."""
        chime = cal.chime()
        with self.lock:
            if self.engine is not None:
                for i, s in enumerate(self.engine.speakers):
                    if s.cfg.get("device") == device:
                        self.engine.inject(i, chime, self.engine.produced + int(0.15 * 48000))
                        return
            if self.cal["running"]:
                raise RuntimeError("Espera a que termine la calibración.")

        def work():
            import sounddevice as sd_
            t = np.arange(len(chime))
            try:
                sd_.play(np.stack([chime, chime], 1), 48000, device=device, blocking=True)
            except Exception as e:
                self.error = f"No pude hacer sonar «{device}»: {e}"
        threading.Thread(target=work, daemon=True).start()

    def finish_setup(self):
        with self.lock:
            self.cfg["onboarded"] = True
            self.save()

    SETTINGS_URLS = {
        "bluetooth": "x-apple.systempreferences:com.apple.BluetoothSettings",
        "sound": "x-apple.systempreferences:com.apple.Sound-Settings.extension",
        "airplay": "x-apple.systempreferences:com.apple.AirDrop-Handoff-Settings.extension",
        "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
        "blackhole": "https://existential.audio/blackhole/",
    }

    def open_settings(self, what: str):
        import subprocess
        url = self.SETTINGS_URLS.get(what)
        if not url:
            raise RuntimeError("Destino desconocido")
        subprocess.run(["open", url], check=False)

    def use_mac_audio(self):
        """Pone la salida de la Mac en el dispositivo virtual (para que Netflix/Spotify pasen por aquí)."""
        lb = coreaudio.loopback_name()
        if not lb:
            raise RuntimeError("Primero instala BlackHole (gratis).")
        coreaudio.set_default_output(lb)

    # --- sala 3D --------------------------------------------------------------------------
    def set_room3d(self, data: dict):
        with self.lock:
            if "height" in data:
                self.cfg["room"]["height"] = round(max(2.0, min(6.0, float(data["height"]))), 2)
            if data.get("shape") in ("rect", "L", "open"):
                self.cfg["room"]["shape"] = data["shape"]
            if isinstance(data.get("cut"), dict):
                c = data["cut"]
                self.cfg["room"]["cut"] = {"corner": c.get("corner", "bl") if c.get("corner") in ("bl", "br", "fl", "fr") else "bl",
                                           "w": round(max(0.5, float(c.get("w", 2))), 2), "d": round(max(0.5, float(c.get("d", 2))), 2)}
            if data.get("open_side") in ("left", "right", "back"):
                self.cfg["room"]["open_side"] = data["open_side"]
            if "decor" in data:
                d = data["decor"]
                dec = self.cfg.setdefault("decor", {"floor": "madera", "walls": "#f2f2f0", "items": []})
                if d.get("floor") in ("madera", "alfombra", "baldosa"):
                    dec["floor"] = d["floor"]
                if isinstance(d.get("walls"), str) and len(d["walls"]) <= 9:
                    dec["walls"] = d["walls"]
                if isinstance(d.get("items"), list):
                    dec["items"] = [{"type": str(i.get("type", "mesa"))[:16], "x": float(i.get("x", 1)),
                                     "y": float(i.get("y", 1)), "rot": float(i.get("rot", 0))}
                                    for i in d["items"][:30]]
            for k, z in (data.get("speaker_z") or {}).items():
                i = int(k)
                if 0 <= i < len(self.cfg["speakers"]):
                    self.cfg["speakers"][i]["z"] = round(max(0.0, min(3.0, float(z))), 2)
            self.save()
            if self.engine is not None:
                self.engine.config["decor"] = self.cfg.get("decor", {})
                self.engine.apply_eq(self.engine.config)

    def advise(self, move_listener: bool, mode: str | None = None) -> dict:
        with self.lock:
            mode = mode if mode in ("envolvente", "frontal") else self.cfg.get("placement_mode", "envolvente")
            self.cfg["placement_mode"] = mode
            room = dict(self.cfg["room"])
            return {"mode": mode,
                    "current": placement.current(room, self.cfg["listener"], self.cfg["speakers"], mode),
                    "best": placement.advise(room, self.cfg["listener"], self.cfg["speakers"], move_listener, mode)}

    def apply_advice(self, prop: dict):
        with self.lock:
            for sp in prop.get("speakers", []):
                i = int(sp["idx"])
                if 0 <= i < len(self.cfg["speakers"]):
                    self.cfg["speakers"][i].update({"x": float(sp["x"]), "y": float(sp["y"]), "z": float(sp.get("z", 1.0))})
            if prop.get("listener"):
                self.cfg["listener"] = {"x": float(prop["listener"]["x"]), "y": float(prop["listener"]["y"])}
            if prop.get("mode") == "envolvente":          # que el ambiente vaya de verdad a los laterales
                snd = dsp.sound_cfg(self.cfg)
                if snd["spatial"] < 2:
                    self.cfg["sound"] = {**snd, "spatial": 2}
                    if self.engine is not None:
                        self.engine.set_sound(self.cfg["sound"])
            self.save()
            if self.engine is not None:
                active = [s.cfg for s in self.engine.speakers]
                by = {s["name"]: s for s in self.cfg["speakers"]}
                for a in active:
                    if a["name"] in by:
                        a.update({k: by[a["name"]][k] for k in ("x", "y", "z") if k in by[a["name"]]})
                self.engine.config["listener"] = self.cfg["listener"]
                self.engine.set_layout(active, self.cfg["listener"])

    def set_sleep(self, minutes: float):
        with self.lock:
            self.sleep_until = time.monotonic() + float(minutes) * 60 if minutes else None
            self.sleep_minutes = float(minutes) if minutes else 0

    # --- perfiles ----------------------------------------------------------------
    def profiles(self) -> list:
        PROFILES.mkdir(exist_ok=True)
        return sorted(p.stem for p in PROFILES.glob("*.json"))

    def profile(self, action: str, name: str):
        name = "".join(c for c in str(name) if c.isalnum() or c in " -_áéíóúñÁÉÍÓÚÑ").strip()[:40]
        if not name:
            raise RuntimeError("Ponle un nombre al perfil.")
        PROFILES.mkdir(exist_ok=True)
        path = PROFILES / f"{name}.json"
        with self.lock:
            if action == "save":
                path.write_text(json.dumps(self.cfg, indent=2, ensure_ascii=False))
            elif action == "load":
                if not path.exists():
                    raise RuntimeError("Ese perfil no existe.")
                self.shutdown()
                self.cfg = json.loads(path.read_text())
                self.cfg["profile"] = name
                self.save()
                self.warm_up()
            elif action == "delete":
                path.unlink(missing_ok=True)
            if action == "save":
                self.cfg["profile"] = name
                self.save()

    def watchdog(self):
        """Cada 3 s: si un parlante se reconectó (ahorro de energía, alcance), macOS le da otro
        identificador y la conexión vieja queda muda aunque figure como conectado. Entonces se reabren
        las salidas y se retoma lo que sonaba."""
        def loop():
            while True:
                time.sleep(3)
                try:
                    su = getattr(self, "sleep_until", None)
                    if su and self.engine is not None and self.mode:
                        left = su - time.monotonic()
                        if left <= 0:
                            self.stop()
                            self.set_volume(self.sleep_vol if hasattr(self, "sleep_vol") else self.volume_db)
                            self.sleep_until = None
                            self.error = "Temporizador: se detuvo la reproducción."
                        elif left <= 30:                                   # últimos 30 s: baja suave
                            if not hasattr(self, "sleep_vol"):
                                self.sleep_vol = self.volume_db
                            self.engine.set_master(self.sleep_vol - 40 * (1 - left / 30))
                except Exception:
                    pass
                try:
                    if self.cal["running"]:
                        continue
                    missing = list(getattr(self, "missing", []) or [])
                    if missing:
                        names = {sp["name"]: sp["device"] for sp in self.cfg["speakers"]}
                        back = coreaudio.output_ids([names[m] for m in missing if m in names])
                        if any(v is not None for v in back.values()):
                            self.reconnect(", ".join(m for m in missing if back.get(names.get(m))))
                            continue
                        if time.monotonic() - getattr(self, "bt_try", 0) > 20:
                            self.bt_try = time.monotonic()
                            self.bt_reconnect([names[m] for m in missing if m in names])
                    if self.engine is None:
                        if self.cfg["speakers"] and time.monotonic() - getattr(self, "idle_try", 0) > 10:
                            self.idle_try = time.monotonic()
                            self.warm_up()
                        continue
                    if not self.engine_ids:
                        continue
                    now_ids = coreaudio.output_ids(list(self.engine_ids))
                    dead = [n for n, i in now_ids.items() if i != self.engine_ids.get(n)]
                    if not dead and self.engine.healthy:
                        continue
                    missing = [n for n in dead if now_ids.get(n) is None]
                    if missing:                       # se desconectó: intentar reconectarlo (cada 20 s)
                        if time.monotonic() - getattr(self, "bt_try", 0) > 20:
                            self.bt_try = time.monotonic()
                            self.error = f"{', '.join(missing)} se desconectó; intentando reconectar…"
                            self.bt_reconnect(missing)
                        continue
                    self.reconnect(", ".join(dead) or "un parlante")
                except Exception as e:
                    self.error = f"Reconexión: {e}"
        threading.Thread(target=loop, daemon=True).start()

    def bt_reconnect(self, names):
        """Parlantes emparejados que se desconectaron (se durmieron, salieron de alcance): si siguen
        encendidos, se reconectan solos con bin/btconnect (IOBluetooth)."""
        import subprocess
        tool = BIN / "btconnect"
        if not tool.exists():
            return
        if not getattr(self, "bt_addr", None):
            try:
                out = subprocess.run(["system_profiler", "-json", "SPBluetoothDataType"], capture_output=True,
                                     text=True, timeout=20).stdout
                data = json.loads(out)["SPBluetoothDataType"][0]
                addr = {}
                for sec in ("device_connected", "device_not_connected"):
                    for item in data.get(sec, []):
                        for name, info in item.items():
                            addr[name] = info.get("device_address")
                self.bt_addr = addr
            except Exception:
                self.bt_addr = {}
        for n in names:
            a = self.bt_addr.get(n)
            if a:
                # Sin --reset: un parlante recién conectado tarda unos segundos en aparecer como salida de
                # audio, y en esa ventana «resetearlo» lo desconectaba (algunos parlantes BT se caían al conectarlos).
                subprocess.run([str(tool), a], capture_output=True, timeout=30)

    def reconnect(self, who: str):
        with self.lock:
            mode, file = self.mode, self.file
            pos = getattr(self.engine.source, "position", None) if self.engine else None
            paused = self.engine.paused if self.engine else False
            if self.engine is not None:
                self.engine.source = SilenceSource()      # que el cierre no se lleve la fuente
            self.shutdown()
            self.engine_error = ""
            self.last_open_try = 0.0
            try:
                if mode and mode != "test":
                    self.start(mode, file)
                    if mode == "file" and pos:
                        self.seek(pos)
                        self.engine.paused = paused
                else:
                    self.ensure_engine()
                self.error = f"{who} se reconectó; listo."
            except Exception as e:
                self.error = f"{who} se reconectó pero no pude reabrirlo: {e}"

    def warm_up(self):
        """Abre las salidas en segundo plano (y las deja enviando silencio)."""
        def work():
            try:
                self.ensure_engine()
                if self.error.startswith("Se desconectó"):
                    self.error = ""
            except Exception as e:
                self.error = str(e)
        threading.Thread(target=work, daemon=True).start()

    # --- cuarto y parlantes ------------------------------------------------------
    def set_layout(self, data: dict):
        with self.lock:
            if "room" in data:
                self.cfg["room"] = {k: max(1.5, min(20.0, float(v))) for k, v in data["room"].items()}
            if "listener" in data:
                self.cfg["listener"] = {k: float(v) for k, v in data["listener"].items()}
            if "mic_pos" in data:
                mp = data["mic_pos"]
                self.cfg["mic_pos"] = None if mp is None else {k: float(v) for k, v in mp.items()}
            for i, pos in enumerate(data.get("speakers", [])):
                if i < len(self.cfg["speakers"]):
                    self.cfg["speakers"][i]["x"] = float(pos["x"])
                    self.cfg["speakers"][i]["y"] = float(pos["y"])
            self.save()
            if self.engine:
                active = [s.cfg for s in self.engine.speakers]
                by_name = {s["name"]: s for s in self.cfg["speakers"]}
                for a in active:
                    if a["name"] in by_name:
                        a["x"], a["y"] = by_name[a["name"]]["x"], by_name[a["name"]]["y"]
                self.engine.set_layout(active, self.cfg["listener"])

    def add_speaker(self, device: str):
        if is_virtual(device):
            raise RuntimeError(f"{device} es el canal por donde entra el audio de la Mac; no puede ser un parlante.")
        with self.lock:
            self.shutdown()
            names = {s["name"] for s in self.cfg["speakers"]}
            name = device
            k = 2
            while name in names:
                name = f"{device} {k}"
                k += 1
            room, n = self.cfg["room"], len(self.cfg["speakers"])
            spots = [(0.2, 0.92), (0.8, 0.92), (0.15, 0.25), (0.85, 0.25), (0.5, 0.95), (0.5, 0.08)]
            fx, fy = spots[n % len(spots)]
            self.cfg["speakers"].append({"name": name, "device": device,
                                         "x": round(room["width"] * fx, 2), "y": round(room["depth"] * fy, 2),
                                         "delay_ms": 0.0, "gain_db": 0.0})
            self.save()
            self.warm_up()

    def partners(self, idx: int) -> list:
        pairs = eqmod.find_pairs(self.cfg["speakers"], self.cfg["listener"])
        return [j for a, b in pairs for i, j in ((a, b), (b, a)) if i == idx]

    def apply_trim(self, idx: int):
        spk = self.cfg["speakers"][idx]
        if self.engine:
            for s in self.engine.speakers:
                if s.name == spk["name"]:
                    s.trim_db = float(spk.get("trim_db", 0.0))
                    s.gain = 10 ** ((s.cal_gain_db + s.trim_db) / 20)

    def even_pairs(self):
        """Iguala el ajuste de volumen de cada pareja (al promedio de los dos)."""
        for a, b in eqmod.find_pairs(self.cfg["speakers"], self.cfg["listener"]):
            sa, sb = self.cfg["speakers"][a], self.cfg["speakers"][b]
            t = round((float(sa.get("trim_db", 0.0)) + float(sb.get("trim_db", 0.0))) / 2, 1)
            sa["trim_db"] = sb["trim_db"] = t
            self.apply_trim(a)
            self.apply_trim(b)

    def update_speaker(self, idx: int, data: dict):
        with self.lock:
            spk = self.cfg["speakers"][idx]
            if data.get("remove"):
                self.shutdown()
                self.cfg["speakers"].pop(idx)
                self.save()
                self.warm_up()
                return
            if "mute" in data:
                spk["mute"] = bool(data["mute"])
                if self.engine:
                    for s in self.engine.speakers:
                        if s.name == spk["name"]:
                            s.mute = spk["mute"]
            if "model" in data:
                if data["model"] in library.BY_ID:
                    spk["model"] = data["model"]
                else:
                    spk.pop("model", None)
            if "trim_db" in data:
                spk["trim_db"] = round(max(-30.0, min(6.0, float(data["trim_db"]))), 1)
                if dsp.sound_cfg(self.cfg).get("pair_volume", True):
                    for k in self.partners(idx):          # su pareja lo acompaña
                        self.cfg["speakers"][k]["trim_db"] = spk["trim_db"]
                        self.apply_trim(k)
                if self.engine:
                    for s in self.engine.speakers:
                        if s.name == spk["name"]:
                            s.trim_db = spk["trim_db"]
                            s.gain = 10 ** ((s.cal_gain_db + s.trim_db) / 20)
            if "gain_db" in data:
                spk["gain_db"] = round(max(-24.0, min(6.0, float(data["gain_db"]))), 1)
                if self.engine:
                    for s in self.engine.speakers:
                        if s.name == spk["name"]:
                            s.gain = 10 ** (spk["gain_db"] / 20)
            if "name" in data and str(data["name"]).strip():
                spk["name"] = str(data["name"]).strip()[:40]
            self.save()

    def set_eq(self, data: dict):
        with self.lock:
            if "preset" in data and data["preset"] in eqmod.PRESETS:
                self.cfg["eq"] = {"preset": data["preset"], "bands": list(eqmod.PRESETS[data["preset"]])}
            if "bands" in data:
                b = [round(max(-12.0, min(12.0, float(x))), 1) for x in data["bands"]][:len(eqmod.BANDS)]
                preset = next((k for k, v in eqmod.PRESETS.items() if v == b), "custom")
                self.cfg["eq"] = {"preset": preset, "bands": b}
            if "match" in data:
                self.cfg["eq_match"] = bool(data["match"])
            if data.get("reset_auto"):
                for spk in self.cfg["speakers"]:
                    spk.pop("eq_auto", None)
            self.save()
            if self.engine:
                if data.get("reset_auto"):
                    for s in self.engine.speakers:
                        s.cfg.pop("eq_auto", None)
                self.engine.apply_eq(self.cfg)

    def fix_input(self, log=None):
        try:
            msg = coreaudio.use_builtin_mic([sp["device"] for sp in self.cfg["speakers"]])
            if msg and log:
                log(f"Entrada de la Mac cambiada para no poner un parlante en modo llamada: {msg}")
        except Exception:
            pass

    def level_hardware(self, log=None) -> None:
        """Todos los parlantes al 100 % de su volumen propio: así parten iguales y la app pone el resto.
        (macOS guarda un volumen por parlante; uno al 50 % suena ~20 dB más bajo que otro al 100 %.)"""
        for spk in self.cfg["speakers"]:
            try:
                before = coreaudio.get_volume(spk["device"])
                if before is not None and before < 0.999:
                    coreaudio.set_volume(spk["device"], 1.0)
                    if log:
                        log(f"Volumen propio de {spk['name']}: {before * 100:.0f} % → 100 %")
            except Exception:
                pass

    # --- calibración -----------------------------------------------------------
    def calibrate(self, rotate: bool = True):
        with self.lock:
            if self.cal["running"]:
                return
            self.stop()
            self.cfg["cal_rotate"] = bool(rotate)
            self.save()
            self.cal = {"running": True, "log": ["Preparando parlantes…"], "done_at": None, "waiting": None}
            self.cal_continue.clear()
            cal.CANCEL.clear()

        def ask(msg):
            self.cal["waiting"] = msg
            self.cal["log"].append(msg)
            ok = False
            for _ in range(300):
                if self.cal_continue.wait(timeout=1) or cal.CANCEL.is_set():
                    ok = not cal.CANCEL.is_set()
                    break
            if cal.CANCEL.is_set():
                self.cal["waiting"] = None
                raise cal.Cancelled()
            self.cal["waiting"] = None
            self.cal_continue.clear()
            if not ok:
                raise RuntimeError("Nadie confirmó el giro en 5 minutos; calibración cancelada.")

        def work():
            try:
                self.level_hardware(self.cal["log"].append)
                eng = self.ensure_engine()
                cfg = cal.run(json.loads(json.dumps(self.cfg)), log=self.cal["log"].append,
                              ask=ask if rotate else None, engine=eng)
                with self.lock:
                    for spk, new in zip(self.cfg["speakers"], cfg["speakers"]):
                        spk["delay_ms"], spk["gain_db"] = new["delay_ms"], new["gain_db"]
                        if "eq_auto" in new:
                            spk["eq_auto"] = new["eq_auto"]
                        for k in ("bass_db", "low_hz", "tf_db"):
                            if k in new:
                                spk[k] = new[k]
                    if "acoustic_latency_ms" in cfg:
                        self.cfg["acoustic_latency_ms"] = cfg["acoustic_latency_ms"]
                    # el motor sigue abierto: devolverle el ajuste del usuario encima de la calibración
                    for s_, spk in zip(eng.speakers, self.cfg["speakers"]):
                        s_.cfg.update({k: spk[k] for k in ("eq_auto", "bass_db", "low_hz", "tf_db") if k in spk})
                        s_.cal_gain_db = float(spk.get("gain_db", 0.0))
                        s_.trim_db = float(spk.get("trim_db", 0.0))
                        s_.gain = 10 ** ((s_.cal_gain_db + s_.trim_db) / 20)
                        ph = eqmod.protect_hz(s_.cfg)
                        s_.protect = dsp.lr4("highpass", ph, 1) if ph else None
                        s_.protect_hz = ph
                    eng.set_sound(dsp.sound_cfg(self.cfg))
                    if "sound" in cfg:
                        self.cfg["sound"] = cfg["sound"]
                    if "calibration_report" in cfg:
                        self.cfg["calibration_report"] = cfg["calibration_report"]
                    self.save()
                self.cal["log"].append("Listo. Retrasos y volúmenes guardados.")
            except cal.Cancelled:
                self.cal["log"].append("Calibración cancelada; no se guardó nada.")
            except Exception as e:  # se muestra en la interfaz
                self.cal["log"].append(f"Error: {e}")
            finally:
                self.cal["running"] = False
                self.cal["done_at"] = time.time()

        threading.Thread(target=work, daemon=True).start()

    # --- estado ----------------------------------------------------------------
    def state(self) -> dict:
        # Si otro hilo está ocupado (p. ej. reconectando un parlante), no colgar la página: último estado.
        if not self.lock.acquire(timeout=2):
            return {**getattr(self, "last_state", {}), "busy": True}
        try:
            st = self._state()
            self.last_state = st
            return st
        finally:
            self.lock.release()

    def _state(self) -> dict:
        with self.lock:
            if self.engine is not None and self.engine.ended:   # el archivo terminó
                self.engine.ended = False
                self.stop()
            if (self.engine is not None and not self.engine.healthy and not self.cal["running"]):
                self.shutdown()                                   # se desconectó un parlante
                self.error = "Se desconectó un parlante; reconectando…"
                self.warm_up()
            if self.monitor is not None and self.monitor.sync.changed:     # guardar lo que corrigió solo
                self.monitor.sync.changed = False
                for sp, es in zip(self.cfg["speakers"], self.engine.speakers):
                    sp["delay_ms"] = round(es.delay / 48000 * 1000, 2)
                self.save()
            status = self.engine.status() if self.engine else []
            live = {s["name"]: s for s in status}
            lst = self.cfg["listener"]
            speakers = []
            for s in self.cfg["speakers"]:
                speakers.append({**s, "spec": placement.spec_for(s),
                                 "azimuth": round(render.speaker_azimuth(s, lst), 1),
                                 "distance": round(render.speaker_distance(s, lst), 2),
                                 "live": live.get(s["name"])})
            matrix = []
            if self.cfg["speakers"]:
                m = render.build_matrix(self.cfg["speakers"], lst)
                matrix = [[round(float(v), 3) for v in row] for row in m]
            return {"room": self.cfg["room"], "listener": lst, "speakers": speakers,
                    "channels": render.CHANNELS, "matrix": matrix,
                    "playing": self.mode, "file": self.file, "volume_db": self.volume_db,
                    "channel": self.channel, "error": self.error,
                    "report": self.cfg.get("calibration_report"),
                    "input_peak": round(self.engine.in_peak, 4) if self.engine else 0.0,
                    "eq": self.cfg["eq"], "eq_match": self.cfg["eq_match"], "bands": eqmod.BANDS,
                    "mic_pos": self.cfg.get("mic_pos"), "cal_rotate": self.cfg.get("cal_rotate", True),
                    "sound": dsp.sound_cfg(self.cfg),
                    "bass_speakers": [self.cfg["speakers"][i]["name"]
                                      for i in dsp.bass_speakers(self.cfg["speakers"])],
                    "progress": ({"pos": round(self.engine.source.position, 1),
                                  "audible": round(self.engine.source.position - self.av_latency(), 3),
                                  "video": Path(self.file).suffix.lower() in VIDEO_EXT,
                                  "dur": round(self.engine.source.duration, 1),
                                  "paused": self.engine.paused}
                                 if self.engine and self.mode == "file" else None),
                    "profiles": self.profiles(), "profile": self.cfg.get("profile"),
                    "scene": self.cfg.get("scene"), "av_offset_ms": self.cfg.get("av_offset_ms", 0),
                    "pairs": [list(p) for p in eqmod.find_pairs(self.cfg["speakers"], self.cfg["listener"])],
                    "object": ({"az": round(self.engine.source.objects[0].az, 1),
                                "dist": round(self.engine.source.objects[0].dist, 2),
                                "el": round(getattr(self.engine.source.objects[0], "el", 0.0), 1),
                                "orbit": self.engine.source.objects[0].orbit,
                                "kind": self.engine.source.objects[0].kind}
                               if self.engine and getattr(self.engine.source, "objects", None) else None),
                    "monitor": dict(self.monitor.stats) if self.monitor else None,
                    "missing": list(getattr(self, "missing", []) or []),
                    "onboarded": bool(self.cfg.get("onboarded")),
                    "decor": self.cfg.get("decor", {"floor": "madera", "walls": "#f2f2f0", "items": []}),
                    "sleep_left": (max(0, int(self.sleep_until - time.monotonic()))
                                   if getattr(self, "sleep_until", None) else 0),
                    "spectrum": self.engine.spectrum() if self.engine is not None and self.mode else None,
                    "room_ready": all(sp.get("tf_db") for sp in self.cfg["speakers"]) and bool(self.cfg["speakers"]),
                    "calibration": {"running": self.cal["running"], "log": self.cal["log"][-12:],
                                    "waiting": self.cal.get("waiting"),
                                    "done_at": self.cal["done_at"]}}


def check_live_output(lb: str = "BlackHole"):
    """La Mac debe sonar hacia BlackHole; si es así, su volumen va al 100 % (BlackHole no se oye directo,
    el volumen real lo pone la app). Con el volumen de la Mac bajo, el audio llega atenuado."""
    import subprocess
    out = coreaudio.default_output_name()   # directo de CoreAudio: no hace falta reiniciar nada
    if lb.lower() not in out.lower():
        raise RuntimeError(f"La salida de la Mac es «{out}». Cámbiala a «{lb}» en Ajustes → Sonido → Salida "
                           f"(o toca «Usar audio de la Mac» en la app).")
    subprocess.run(["osascript", "-e", "set volume output volume 100 without output muted"], check=False)


def is_virtual(device: str) -> bool:
    if "blackhole" in device.lower() or "soundflower" in device.lower():
        return True
    return any(d["name"] == device and d["kind"] in ("virtual", "aggregate") for d in coreaudio.list_devices())


def list_outputs(configured: set) -> list:
    """Salidas para agregar, leídas en vivo de CoreAudio: la lista de PortAudio se congela mientras el
    motor está abierto, así que un parlante conectado con la app abierta no aparecía."""
    out = []
    for d in coreaudio.list_devices():
        if d["output"] and d["kind"] not in ("virtual", "aggregate") and "salida múltiple" not in d["name"].lower() \
                and "multi-output" not in d["name"].lower():
            out.append({"name": d["name"], "used": d["name"] in configured})
    out.sort(key=lambda d: d["used"])
    return out


def list_media() -> list:
    found = []
    for base in ("Movies", "Music", "Downloads", "Desktop"):
        root = Path.home() / base
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if len(found) >= 60:
                    break
                if p.suffix.lower() in MEDIA_EXT and p.is_file() and "/." not in str(p):
                    found.append({"path": str(p), "name": p.name, "folder": base,
                                  "mtime": p.stat().st_mtime})
        except (PermissionError, OSError):
            continue
    found.sort(key=lambda f: -f["mtime"])
    return found[:40]


def lan_ip() -> str:
    """IP de esta Mac en la Wi-Fi/Ethernet local (privada). Ignora VPN y Tailscale (100.64/10, utun):
    el iPhone en la misma Wi-Fi llega directo, sin instalar nada."""
    import ipaddress
    import subprocess
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        out = ""
    iface, cands = "", []
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":")[0]
        elif "inet " in line and iface.startswith(("en", "bridge")):
            ip = line.split()[1]
            a = ipaddress.ip_address(ip)
            if a.is_private and not a.is_loopback and a not in ipaddress.ip_network("100.64.0.0/10"):
                cands.append((iface != "en0", iface, ip))     # en0 (Wi-Fi) primero
    return sorted(cands)[0][2] if cands else "127.0.0.1"


def computer_name() -> str:
    import subprocess
    try:
        return subprocess.run(["scutil", "--get", "ComputerName"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "esta Mac"


def local_hostname() -> str | None:
    """Nombre Bonjour de esta Mac (p. ej. «Mi-Mac.local»): apunta a esta Mac y no a otra."""
    import subprocess
    try:
        n = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{n}.local" if n else None
    except Exception:
        return None


PORT = 8890


def remote_url(prefer_ip: bool = True) -> str:
    host = lan_ip() if prefer_ip else (local_hostname() or lan_ip())
    return f"http://{host}:{PORT}/?t={CTRL.cfg.get('remote_token', '')}"


def remote_qr() -> bytes:
    import io
    import qrcode
    img = qrcode.make(remote_url(), box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


CTRL: Controller | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    # --- acceso desde el iPhone: solo con el código del QR ---------------------------------
    def _local(self) -> bool:
        return self.client_address[0] in ("127.0.0.1", "::1")

    def _authorized(self) -> bool:
        if self._local():
            return True
        tok = CTRL.cfg.get("remote_token", "")
        cookie = self.headers.get("Cookie", "")
        return bool(tok) and f"sbt={tok}" in cookie.replace(" ", "")

    def _deny(self):
        body = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                "<body style='font:17px -apple-system;padding:32px;background:#f5f5f7;color:#1d1d1f'>"
                "<h2>Surround BT</h2><p>Para controlar desde aquí, escanea el código QR que muestra la app "
                "en la Mac (tarjeta «Control desde el iPhone»).</p>").encode()
        self._send(403, body, "text/html; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?")[0]
        query = self.path.partition("?")[2]
        if path == "/" and "t=" in query and not self._local():
            tok = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv).get("t", "")
            if tok and tok == CTRL.cfg.get("remote_token"):
                self.send_response(302)
                self.send_header("Set-Cookie", f"sbt={tok}; Path=/; Max-Age=31536000; SameSite=Strict")
                self.send_header("Location", "/")
                self.end_headers()
                return
        if path == "/room3d.js" or path.startswith("/vendor/"):
            f = (UI / path.lstrip("/")).resolve()
            if UI.resolve() in f.parents and f.is_file():          # nada fuera de la carpeta de la interfaz
                ct = "text/javascript" if f.suffix == ".js" else "text/plain; charset=utf-8"
                return self._send(200, f.read_bytes(), ct)
            return self._send(404, b"{}")
        static = {"/manifest.json": ("manifest.json", "application/manifest+json"),
                  "/icon-180.png": ("icon-180.png", "image/png"), "/icon-512.png": ("icon-512.png", "image/png")}
        if path in static:
            f, ct = static[path]
            return self._send(200, (UI / f).read_bytes(), ct)
        if not self._authorized():
            return self._deny()
        if path == "/qr.png":
            return self._send(200, remote_qr(), "image/png")
        if path == "/api/remote":
            return self._json({"url": remote_url(), "url_name": remote_url(False), "ip": lan_ip(),
                               "host": local_hostname(), "computer": computer_name()})
        if path in ("/", "/index.html"):
            return self._send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._json(CTRL.state())
        if path == "/api/library":
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv).get("q", "")
            from urllib.parse import unquote_plus
            return self._json([{k: e[k] for k in ("id", "brand", "model", "size", "disp", "low_hz")}
                               for e in library.search(unquote_plus(q), 40)])
        if path == "/api/setup":
            return self._json(CTRL.setup_info())
        if path == "/api/devices":
            return self._json(list_outputs({s["device"] for s in CTRL.cfg["speakers"]}))
        if path == "/media":
            return self._serve_media()
        if path == "/api/media":
            return self._json(list_media())
        self._send(404, b"{}")

    def _serve_media(self):
        """Solo el archivo que se está reproduciendo (nada más del disco)."""
        f = CTRL.file if CTRL.mode == "file" else None
        if not f or not Path(f).expanduser().is_file():
            return self._send(404, b"{}")
        p = Path(f).expanduser()
        size = p.stat().st_size
        ctype = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
                 ".mkv": "video/x-matroska"}.get(p.suffix.lower(), "application/octet-stream")
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if (a and b) else size - 1
        end = min(end, start + 4 * 1024 * 1024 - 1, size - 1)    # trozos de 4 MB
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as fh:
            fh.seek(start)
            try:
                self.wfile.write(fh.read(end - start + 1))
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        if not self._authorized():
            return self._json({"error": "Sin acceso"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "JSON inválido"}, 400)
        path = self.path
        try:
            if path == "/api/start":
                CTRL.start(data.get("mode", "test"), data.get("file"))
            elif path == "/api/stop":
                CTRL.stop()
            elif path == "/api/volume":
                CTRL.set_volume(data["db"])
            elif path == "/api/layout":
                CTRL.set_layout(data)
            elif path == "/api/speakers/add":
                CTRL.add_speaker(data["device"])
            elif path.startswith("/api/speakers/"):
                CTRL.update_speaker(int(path.rsplit("/", 1)[1]), data)
            elif path == "/api/eq":
                CTRL.set_eq(data)
            elif path == "/api/sound":
                CTRL.set_sound(data)
            elif path == "/api/setup/speakers":
                CTRL.setup_speakers(data.get("devices", []), data.get("layout", ""))
            elif path == "/api/setup/identify":
                CTRL.identify_device(data.get("device", ""))
            elif path == "/api/setup/done":
                CTRL.finish_setup()
            elif path == "/api/open":
                CTRL.open_settings(data.get("what", ""))
            elif path == "/api/use-mac-audio":
                CTRL.use_mac_audio()
            elif path == "/api/placement":
                return self._json(CTRL.advise(bool(data.get("move_listener")), data.get("mode")))
            elif path == "/api/placement/apply":
                CTRL.apply_advice(data)
            elif path == "/api/room3d":
                CTRL.set_room3d(data)
            elif path == "/api/sleep":
                CTRL.set_sleep(data.get("minutes", 0))
            elif path == "/api/avoffset":
                CTRL.set_av_offset(data.get("ms", 0))
            elif path == "/api/scene":
                CTRL.set_scene(data.get("name", ""))
            elif path == "/api/object":
                CTRL.set_object(data)
            elif path == "/api/pause":
                CTRL.pause(data.get("paused", True))
            elif path == "/api/seek":
                CTRL.seek(data["pos"])
            elif path.startswith("/api/identify/"):
                CTRL.identify(int(path.rsplit("/", 1)[1]))
            elif path == "/api/sync":
                CTRL.quick_sync()
            elif path == "/api/profile":
                CTRL.profile(data.get("action", "save"), data.get("name", ""))
            elif path == "/api/calibrate/cancel":
                cal.CANCEL.set()
                CTRL.cal_continue.set()
            elif path == "/api/calibrate/continue":
                CTRL.cal_continue.set()
            elif path == "/api/calibrate":
                CTRL.calibrate(bool(data.get("rotate", True)))
            else:
                return self._json({"error": "no existe"}, 404)
        except Exception as e:
            CTRL.error = str(e)
            return self._json({"error": str(e)}, 400)
        self._json(CTRL.state())


def serve(host="0.0.0.0", port=8890):
    global CTRL, PORT
    PORT = port
    CTRL = Controller()
    CTRL.warm_up()          # conecta los parlantes al abrir la app; así el Bluetooth ya está asentado
    CTRL.watchdog()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Surround BT en http://{'localhost' if host == '127.0.0.1' else host}:{port}  (Ctrl+C para salir)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CTRL.stop()
