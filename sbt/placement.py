"""Dónde poner cada parlante. Recibe la forma de la sala, tu lugar y los modelos (biblioteca o lo medido),
y propone ubicación, altura y orientación con un puntaje de 0 a 100.

Criterios (cada uno 0–100):
- Ángulos: qué tan cerca quedan del reparto surround ideal para esa cantidad de parlantes (ITU-R BS.775).
- Equilibrio: distancias parejas a ti (y las parejas izquierda/derecha simétricas).
- Graves: un parlante chico gana graves pegado a una pared (+3 dB); uno grande retumba en una esquina.
  Además, tu lugar no debería quedar en la mitad exacta del largo (ahí se cancelan graves) ni pegado a la pared.
- Volumen: que cada uno alcance ~85 dB donde estás (con margen para picos).
"""
from __future__ import annotations

import itertools
import math

from . import library

ROLES = {
    1: [("C", 0)],
    2: [("L", -30), ("R", 30)],
    3: [("L", -30), ("C", 0), ("R", 30)],
    4: [("L", -30), ("R", 30), ("Ls", -110), ("Rs", 110)],
    5: [("L", -30), ("C", 0), ("R", 30), ("Ls", -110), ("Rs", 110)],
    6: [("L", -30), ("C", 0), ("R", 30), ("Ls", -110), ("Rs", 110), ("Cb", 180)],
    7: [("L", -30), ("C", 0), ("R", 30), ("Ls", -90), ("Rs", 90), ("Lb", -145), ("Rb", 145)],
}
ROLE_NAME = {"L": "frente izquierdo", "R": "frente derecho", "C": "centro", "Ls": "lateral izquierdo",
             "Rs": "lateral derecho", "Cb": "atrás al centro", "Lb": "atrás izquierdo", "Rb": "atrás derecho"}
WEIGHTS = {"angles": 0.35, "balance": 0.2, "bass": 0.2, "volume": 0.25}
MARGIN = 0.25


# Envolvente: con pocos parlantes, priorizar que el sonido te rodee (laterales un poco hacia atrás)
# en vez de un frente de cine. Con 4 o más ya hay frente y atrás, igual que el estándar.
ROLES_ENV = {
    1: [("C", 0)],
    2: [("Ls", -95), ("Rs", 95)],
    3: [("C", 0), ("Ls", -100), ("Rs", 100)],
}


def roles_for(n: int, mode: str = "envolvente") -> list:
    if mode == "envolvente" and n in ROLES_ENV:
        return ROLES_ENV[n]
    if n in ROLES:
        return ROLES[n]
    return [(f"S{i}", round(-180 + 360 * i / n + 180 / n)) for i in range(n)]


def spec_for(spk: dict) -> dict:
    """Ficha del parlante: el modelo elegido (o detectado) y, si hay, lo medido al calibrar."""
    e = library.BY_ID.get(spk.get("model") or "") or library.match(spk.get("name", "")) \
        or library.BY_ID["generic|parlante pequeño"]
    s = {"model": f"{e['brand']} {e['model']}".replace("Generic ", ""), "id": e["id"], "size": e["size"],
         "low_hz": e["low_hz"], "spl": e["spl"], "disp": e["disp"], "measured": False}
    if spk.get("low_hz") and spk.get("bass_db") is not None:      # medido de verdad: manda
        s["low_hz"] = float(spk["low_hz"])
        s["measured"] = True
    return s


# --- geometría de la sala --------------------------------------------------------------------
def _cut_rect(room: dict):
    if room.get("shape") != "L":
        return None
    c = room.get("cut") or {}
    W, D = room["width"], room["depth"]
    w, d = min(W * 0.8, float(c.get("w", W / 2))), min(D * 0.8, float(c.get("d", D / 2)))
    corner = c.get("corner", "bl")       # b = lado de atrás (y bajo), f = lado de la pantalla (y alto)
    x0 = 0.0 if corner.endswith("l") else W - w
    y0 = 0.0 if corner.startswith("b") else D - d
    return (x0, y0, x0 + w, y0 + d)


def inside(room: dict, x: float, y: float, m: float = MARGIN) -> bool:
    if not (m <= x <= room["width"] - m and m <= y <= room["depth"] - m):
        return False
    c = _cut_rect(room)
    return not (c and c[0] - m < x < c[2] + m and c[1] - m < y < c[3] + m)


def clamp(room: dict, x: float, y: float) -> tuple:
    W, D = room["width"], room["depth"]
    x, y = min(max(x, MARGIN), W - MARGIN), min(max(y, MARGIN), D - MARGIN)
    c = _cut_rect(room)
    if c and c[0] - MARGIN < x < c[2] + MARGIN and c[1] - MARGIN < y < c[3] + MARGIN:
        # empujar al borde más cercano del recorte
        opts = [(abs(x - (c[0] - MARGIN)), c[0] - MARGIN, y), (abs(x - (c[2] + MARGIN)), c[2] + MARGIN, y),
                (abs(y - (c[1] - MARGIN)), x, c[1] - MARGIN), (abs(y - (c[3] + MARGIN)), x, c[3] + MARGIN)]
        opts = [o for o in opts if MARGIN <= o[1] <= W - MARGIN and MARGIN <= o[2] <= D - MARGIN]
        if opts:
            _, x, y = min(opts)
    return x, y


def wall_dists(room: dict, x: float, y: float) -> list:
    """Distancias a las paredes (sin contar el lado abierto, si la sala es abierta)."""
    W, D = room["width"], room["depth"]
    d = {"left": x, "right": W - x, "back": y, "front": D - y}
    if room.get("shape") == "open":
        d.pop(room.get("open_side", "back"), None)
    out = list(d.values())
    c = _cut_rect(room)
    if c:   # las dos paredes interiores del recorte
        if c[0] <= x <= c[2]:
            out.append(abs(y - (c[3] if y > c[3] else c[1])))
        if c[1] <= y <= c[3]:
            out.append(abs(x - (c[2] if x > c[2] else c[0])))
    return sorted(out)


# --- puntaje ----------------------------------------------------------------------------------
def _az(p, lst):
    return math.degrees(math.atan2(p[0] - lst["x"], p[1] - lst["y"]))


def _wrap(a):
    return (a + 180) % 360 - 180


def score(room: dict, lst: dict, pts: list, specs: list, targets: list, mode: str = "frontal") -> dict:
    n = len(pts)
    dist = [max(0.3, math.hypot(p[0] - lst["x"], p[1] - lst["y"])) for p in pts]
    # ángulos
    errs = [abs(_wrap(_az(p, lst) - t)) for p, t in zip(pts, targets)]
    angles = max(0.0, 100 - 2.2 * (sum(errs) / n)) if n > 1 else 100.0
    # equilibrio (y simetría de parejas)
    mean = sum(dist) / n
    cv = (sum((d - mean) ** 2 for d in dist) / n) ** 0.5 / mean
    pair_pen = 0.0
    for i, j in itertools.combinations(range(n), 2):
        if abs(targets[i] + targets[j]) < 1 and targets[i] != 0:
            pair_pen += abs(dist[i] - dist[j]) / max(dist[i], dist[j]) * 100
    # distancia cómoda: ni encima (escenario diminuto) ni demasiado lejos (se pierde detalle)
    comfort = 0.0
    for d, t in zip(dist, targets):
        lo, hi = (1.2, 2.8) if abs(t) >= 60 else (1.8, 3.2)      # laterales/traseros: más cerca está bien
        comfort += 0 if lo <= d <= hi else (lo - d) * 60 if d < lo else (d - hi) * 25
    comfort /= n
    for i, j in itertools.combinations(range(n), 2):          # izquierda y derecha bien separadas
        if abs(targets[i] + targets[j]) < 1 and targets[i] != 0:
            sep = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            comfort += max(0.0, (1.2 - sep) * 50)
    balance = max(0.0, 100 - (70 if mode == "envolvente" else 220) * cv - pair_pen - comfort)
    # graves
    bass_parts = []
    for p, s in zip(pts, specs):
        w = wall_dists(room, *p)
        small = s["low_hz"] >= 85 or s["size"] in ("xs", "s")
        if small:   # pegado a la pared: refuerzo de graves
            bass_parts.append(100 if w[0] <= 0.35 else max(45, 100 - (w[0] - 0.35) * 45))
        else:       # grande: lejos de esquinas (dos paredes cerca = retumbe)
            corner = w[0] < 0.5 and len(w) > 1 and w[1] < 0.5
            bass_parts.append(55 if corner else (100 if w[0] >= 0.3 else 85))
    rel = lst["y"] / room["depth"]
    seat = 100 - (35 if 0.44 < rel < 0.56 else 0) - (30 if wall_dists(room, lst["x"], lst["y"])[0] < 0.6 else 0)
    bass = 0.75 * (sum(bass_parts) / n) + 0.25 * seat
    # volumen
    vols = []
    for d, p, s in zip(dist, pts, specs):
        gain = 3 if wall_dists(room, *p)[0] <= 0.35 else 0
        at = s["spl"] - 20 * math.log10(d) + gain + 10 * math.log10(n)   # todos juntos suman
        vols.append(max(0.0, min(100.0, (at - 78) / 12 * 100)))
    volume = sum(vols) / n
    # nadie encimado ni fuera de la sala
    clash = sum(1 for i, j in itertools.combinations(range(n), 2)
                if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < 0.5)
    parts = {"angles": angles, "balance": balance, "bass": bass, "volume": volume}
    total = sum(parts[k] * w for k, w in WEIGHTS.items()) - 15 * clash
    total -= sum(10 for p in pts if not inside(room, *p, m=0.05))
    return {"total": round(max(0, min(100, total))), "parts": {k: round(v) for k, v in parts.items()}}


# --- quién va dónde ---------------------------------------------------------------------------
def capability(s: dict) -> float:
    return s["spl"] + (100 - min(150, s["low_hz"])) / 3


def assign_roles(specs: list, mode: str = "envolvente") -> list:
    """Índice de rol para cada parlante. Centro/frente al más capaz; parejas iguales a izquierda/derecha."""
    n = len(specs)
    roles = roles_for(n, mode)
    order = sorted(range(n), key=lambda i: -capability(specs[i]))
    out = [None] * n
    names = [r[0] for r in roles]
    free = list(range(n))
    # parejas de modelos iguales → L/R (o el par simétrico disponible)
    pairs_slots = [(names.index(a), names.index(b)) for a, b in (("L", "R"), ("Ls", "Rs"), ("Lb", "Rb"))
                   if a in names and b in names]
    by_model = {}
    for i in order:
        by_model.setdefault(specs[i]["id"], []).append(i)
    for slots in pairs_slots:
        for mid, idxs in by_model.items():
            avail = [i for i in idxs if out[i] is None]
            if len(avail) >= 2 and all(s in free for s in slots):
                out[avail[0]], out[avail[1]] = slots
                free.remove(slots[0]); free.remove(slots[1])
                break
    # el resto por capacidad: C primero, luego frente, luego atrás
    priority = sorted(free, key=lambda k: (0 if names[k] == "C" else 1 if names[k] in ("L", "R") else 2, k))
    for i in [i for i in order if out[i] is None]:
        out[i] = priority.pop(0)
    return out


def current(room: dict, lst: dict, speakers: list, mode: str = "envolvente") -> dict:
    """Puntaje de cómo están hoy (cada parlante al rol ideal más parecido)."""
    specs = [spec_for(s) for s in speakers]
    pts = [(s["x"], s["y"]) for s in speakers]
    roles = roles_for(len(pts), mode)
    best = None
    for perm in itertools.permutations(range(len(roles))):
        targets = [roles[k][1] for k in perm]
        sc = score(room, lst, pts, specs, targets, mode)
        if not best or sc["total"] > best["total"]:
            best = sc
    return best


def advise(room: dict, lst: dict, speakers: list, move_listener: bool = False, mode: str = "envolvente") -> dict:
    n = len(speakers)
    if n == 0:
        return {"error": "No hay parlantes."}
    room = {**room, "height": room.get("height", 2.5)}
    specs = [spec_for(s) for s in speakers]
    role_idx = assign_roles(specs, mode)
    roles = roles_for(n, mode)
    W, D = room["width"], room["depth"]
    if move_listener:
        # a 1,8–3,4 m de la pantalla, centrado (el puntaje castiga la mitad exacta del largo)
        cands = [(W / 2, D - d) for d in (1.8, 2.1, 2.4, 2.7, 3.0, 3.4)]
        cands = [c for c in cands if inside(room, *c, m=0.6)] or [(lst["x"], lst["y"])]
    else:
        cands = [(lst["x"], lst["y"])]
    best = None
    for lx, ly in cands:
        L = {"x": lx, "y": ly}
        front = [i for i in range(n) if abs(roles[role_idx[i]][1]) < 60]
        has_front, has_side = bool(front), len(front) < n
        r_front_opts = [0.8 + 0.2 * k for k in range(16)] if has_front else [None]
        r_side_opts = [0.8 + 0.2 * k for k in range(14)] if has_side else [None]
        for r_front in r_front_opts:
            for r_side in r_side_opts:
                for front_wall in ((False, True) if has_front else (False,)):
                    pts = []
                    for i in range(n):
                        name, a = roles[role_idx[i]]
                        r = r_front if abs(a) < 60 else r_side      # frente y laterales con distancia propia
                        x = lx + r * math.sin(math.radians(a))
                        y = ly + r * math.cos(math.radians(a))
                        if name == "C" or (front_wall and abs(a) < 60):   # el centro, siempre bajo la pantalla
                            y = D - MARGIN
                            x = lx + (y - ly) * math.tan(math.radians(a))
                        x, y = clamp(room, x, y)
                        # chicos: hacia la pared sobre la misma línea (gana graves), sin alejarlo más de 25 %
                        if (specs[i]["low_hz"] >= 85 or specs[i]["size"] in ("xs", "s")) and name != "C":
                            x0, y0 = x, y
                            for _ in range(6):
                                if wall_dists(room, x, y)[0] <= 0.3:
                                    break
                                nx, ny = lx + (x - lx) * 1.04, ly + (y - ly) * 1.04
                                if not inside(room, nx, ny):
                                    break
                                x, y = nx, ny
                            if wall_dists(room, x, y)[0] > 0.3:
                                x, y = x0, y0
                        pts.append((x, y))
                    targets = [roles[role_idx[i]][1] for i in range(n)]
                    sc = score(room, L, pts, specs, targets, mode)
                    if not best or sc["total"] > best["sc"]["total"]:
                        best = {"sc": sc, "pts": pts, "L": L}
    out = []
    ear = 1.05
    for i, (s, p) in enumerate(zip(speakers, best["pts"])):
        name, a = roles[role_idx[i]]
        sp = specs[i]
        L = best["L"]
        d = math.hypot(p[0] - L["x"], p[1] - L["y"])
        z = ear if name in ("L", "R", "C") else ear + 0.25        # traseros un poco más altos
        w = wall_dists(room, *p)
        why = [f"{ROLE_NAME.get(name, name).capitalize()}, a {abs(round(_az(p, L)))}° y {d:.1f} m de ti."]
        if (sp["low_hz"] >= 85 or sp["size"] in ("xs", "s")) and w[0] <= 0.35:
            why.append("Pegado a la pared: así gana graves (es pequeño).")
        elif sp["size"] in ("l", "xl"):
            why.append("Lejos de las esquinas para que los graves no retumben.")
        why.append(f"A ~{z:.1f} m de altura" + (", girado hacia ti." if sp["disp"] == "frontal" else
                                                  (", con su frente hacia ti." if sp["disp"] == "estereo" else
                                                   "; emite en 360°, la orientación no importa.")))
        out.append({"idx": i, "name": s["name"], "role": name, "role_name": ROLE_NAME.get(name, name),
                    "x": round(p[0], 2), "y": round(p[1], 2), "z": round(z, 2), "model": sp["model"],
                    "measured": sp["measured"], "why": " ".join(why)})
    tips = []
    rel = best["L"]["y"] / D
    if move_listener:
        tips.append(f"Tu lugar ideal: a {D - best['L']['y']:.1f} m de la pantalla, centrado.")
    elif 0.44 < rel < 0.56:
        tips.append("Estás justo en la mitad del largo de la sala: ahí se cancelan graves. Moverte 40–60 cm adelante o atrás ayuda.")
    if n == 2 and mode == "envolvente":
        tips.append("Con 2 parlantes a los lados te rodea el ambiente; un 3.º al frente fija las voces al centro.")
    elif n == 2:
        tips.append("Con 2 parlantes es estéreo: para surround real, un 3.º al centro o 2 atrás.")
    if n == 3 and mode == "envolvente":
        tips.append("Con 3 parlantes: el central lleva voces y graves; los laterales, el ambiente que te rodea.")
    if best["sc"]["parts"]["volume"] < 50:
        tips.append("Tus parlantes se quedan cortos de volumen para esta sala: acércalos a ti o suma uno más grande.")
    return {"mode": mode, "score": best["sc"]["total"], "parts": best["sc"]["parts"], "listener": best["L"],
            "speakers": out, "tips": tips}
