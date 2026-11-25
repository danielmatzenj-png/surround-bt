"""Biblioteca de parlantes Bluetooth populares (valores APROXIMADOS de fichas públicas y reseñas).

Por modelo:
- size: xs (bolsillo) · s (botella) · m (lonchera) · l (grande) · xl (fiesta/torre)
- low_hz: frecuencia más baja útil (≈ −6 dB)
- spl: volumen máximo estimado a 1 m (dB SPL)
- disp: "frontal" (emite hacia adelante), "360" (todas direcciones), "estereo" (dos lados, ancho)
- dims: ancho × alto × fondo (cm)
Cuando la app ya calibró un parlante, lo medido (bass_db, low_hz) reemplaza estos valores.
"""
from __future__ import annotations

import re

_RAW = [
    # marca, modelo, tamaño, low_hz, spl, dispersión, (an, al, fo), alias
    ("JBL", "Go 2", "xs", 180, 80, "frontal", (8.6, 7.1, 3.2), ""),
    ("JBL", "Go 3", "xs", 110, 82, "frontal", (8.8, 7.5, 4.1), ""),
    ("JBL", "Go 4", "xs", 90, 83, "frontal", (9.4, 7.7, 4.2), ""),
    ("JBL", "Clip 3", "s", 120, 84, "frontal", (9.7, 15.0, 4.6), ""),
    ("JBL", "Clip 4", "s", 100, 85, "frontal", (8.6, 13.4, 4.6), ""),
    ("JBL", "Clip 5", "s", 75, 87, "frontal", (9.5, 13.5, 5.4), ""),
    ("JBL", "Flip 4", "s", 70, 88, "frontal", (17.5, 6.8, 7.0), ""),
    ("JBL", "Flip 5", "s", 65, 89, "frontal", (18.1, 6.9, 7.4), ""),
    ("JBL", "Flip 6", "s", 63, 90, "frontal", (17.8, 6.8, 7.2), ""),
    ("JBL", "Flip 7", "s", 60, 91, "frontal", (18.2, 6.9, 7.1), ""),
    ("JBL", "Charge 4", "m", 60, 93, "frontal", (22.0, 9.5, 9.3), ""),
    ("JBL", "Charge 5", "m", 60, 94, "frontal", (22.3, 9.6, 9.4), ""),
    ("JBL", "Charge 6", "m", 55, 95, "frontal", (23.0, 10.0, 9.6), ""),
    ("JBL", "Xtreme 2", "l", 55, 97, "frontal", (28.8, 13.6, 13.2), ""),
    ("JBL", "Xtreme 3", "l", 53, 98, "frontal", (29.7, 13.6, 13.4), ""),
    ("JBL", "Xtreme 4", "l", 50, 99, "frontal", (29.7, 14.1, 13.4), ""),
    ("JBL", "Boombox 2", "xl", 45, 103, "frontal", (48.4, 25.7, 20.0), ""),
    ("JBL", "Boombox 3", "xl", 40, 105, "frontal", (48.2, 25.7, 20.0), ""),
    ("JBL", "PartyBox Encore", "xl", 45, 104, "frontal", (27.6, 32.0, 28.3), ""),
    ("JBL", "PartyBox 110", "xl", 45, 106, "frontal", (28.4, 56.8, 30.4), ""),
    ("JBL", "PartyBox 310", "xl", 40, 109, "frontal", (36.9, 68.9, 34.6), ""),
    ("JBL", "Pulse 4", "m", 70, 90, "360", (9.6, 20.0, 9.6), ""),
    ("JBL", "Pulse 5", "m", 55, 92, "360", (10.1, 21.6, 10.1), ""),
    ("JBL", "Authentics 200", "l", 45, 98, "estereo", (33.4, 18.7, 15.8), ""),
    ("Bose", "SoundLink Mini", "s", 60, 88, "estereo", (18.0, 5.1, 5.8), "soundlink mini, mini soundlink"),
    ("Bose", "SoundLink Mini II", "s", 60, 89, "estereo", (18.0, 5.8, 5.1), "mini soundlink ii"),
    ("Bose", "SoundLink Micro", "xs", 110, 84, "frontal", (9.8, 9.8, 3.5), ""),
    ("Bose", "SoundLink Flex", "s", 70, 89, "frontal", (20.1, 9.0, 5.2), ""),
    ("Bose", "SoundLink Flex (2.ª gen)", "s", 65, 90, "frontal", (20.1, 9.0, 5.2), "flex ii"),
    ("Bose", "SoundLink Color II", "s", 80, 87, "frontal", (12.7, 13.1, 5.6), "color ii"),
    ("Bose", "SoundLink Revolve", "m", 60, 89, "360", (8.2, 15.2, 8.2), ""),
    ("Bose", "SoundLink Revolve II", "m", 58, 90, "360", (8.2, 15.2, 8.2), ""),
    ("Bose", "SoundLink Revolve+", "m", 50, 93, "360", (10.5, 18.4, 10.5), "revolve plus"),
    ("Bose", "SoundLink Revolve+ II", "m", 48, 94, "360", (10.5, 18.4, 10.5), "revolve plus ii"),
    ("Bose", "SoundLink Max", "l", 45, 98, "estereo", (26.6, 10.5, 12.0), ""),
    ("Bose", "SoundLink Home", "m", 55, 91, "estereo", (18.8, 13.8, 5.0), ""),
    ("Bose", "Home Speaker 500", "l", 45, 96, "estereo", (17.0, 20.3, 10.9), ""),
    ("Sony", "SRS-XB12", "xs", 100, 83, "frontal", (7.5, 9.5, 7.5), ""),
    ("Sony", "SRS-XB13", "xs", 95, 84, "frontal", (7.6, 9.5, 7.6), ""),
    ("Sony", "SRS-XB23", "s", 70, 88, "frontal", (7.6, 21.8, 7.6), ""),
    ("Sony", "SRS-XB33", "m", 60, 92, "frontal", (24.6, 9.7, 9.8), ""),
    ("Sony", "SRS-XB43", "l", 50, 96, "frontal", (30.5, 11.6, 12.0), ""),
    ("Sony", "SRS-XE200", "s", 65, 89, "360", (8.6, 20.4, 8.8), ""),
    ("Sony", "SRS-XE300", "m", 55, 92, "360", (10.4, 23.4, 10.4), ""),
    ("Sony", "SRS-XG300", "l", 45, 98, "estereo", (32.4, 13.4, 13.6), ""),
    ("Sony", "SRS-XG500", "xl", 40, 102, "estereo", (46.0, 28.0, 19.7), ""),
    ("Sony", "ULT Field 1", "s", 70, 88, "frontal", (20.6, 7.7, 7.6), ""),
    ("Sony", "ULT Field 3", "m", 55, 93, "frontal", (27.4, 10.9, 10.0), ""),
    ("Sony", "ULT Field 7", "xl", 40, 102, "estereo", (59.4, 26.5, 22.1), ""),
    ("Ultimate Ears", "Wonderboom 2", "s", 75, 87, "360", (9.5, 10.4, 9.5), "ue wonderboom 2"),
    ("Ultimate Ears", "Wonderboom 3", "s", 75, 87, "360", (9.5, 10.4, 9.5), "ue wonderboom 3"),
    ("Ultimate Ears", "Wonderboom 4", "s", 72, 88, "360", (9.5, 10.4, 9.5), "ue wonderboom 4"),
    ("Ultimate Ears", "Boom 3", "m", 60, 90, "360", (7.3, 18.4, 7.3), "ue boom 3"),
    ("Ultimate Ears", "Boom 4", "m", 58, 91, "360", (7.3, 18.4, 7.3), "ue boom 4"),
    ("Ultimate Ears", "Megaboom 3", "m", 52, 93, "360", (8.7, 22.5, 8.7), "ue megaboom 3"),
    ("Ultimate Ears", "Megaboom 4", "m", 50, 94, "360", (8.7, 22.5, 8.7), "ue megaboom 4"),
    ("Ultimate Ears", "Everboom", "m", 55, 94, "360", (10.0, 19.0, 10.0), "ue everboom"),
    ("Ultimate Ears", "Epicboom", "l", 45, 98, "360", (11.4, 23.3, 10.8), "ue epicboom"),
    ("Ultimate Ears", "Hyperboom", "xl", 40, 104, "estereo", (19.0, 36.4, 18.5), "ue hyperboom"),
    ("Anker Soundcore", "Mini 3", "xs", 110, 82, "frontal", (6.8, 8.9, 6.8), "soundcore mini"),
    ("Anker Soundcore", "Soundcore 2", "s", 70, 88, "frontal", (16.7, 5.6, 4.8), "soundcore 2"),
    ("Anker Soundcore", "Soundcore 3", "s", 65, 89, "frontal", (17.4, 5.7, 5.9), "soundcore 3"),
    ("Anker Soundcore", "Motion 100", "s", 65, 88, "estereo", (19.5, 5.4, 5.0), "motion 100"),
    ("Anker Soundcore", "Motion 300", "s", 60, 89, "estereo", (19.6, 9.2, 6.0), "motion 300"),
    ("Anker Soundcore", "Motion+", "m", 50, 93, "estereo", (25.5, 8.0, 7.9), "motion plus, motion+"),
    ("Anker Soundcore", "Motion X600", "l", 45, 97, "estereo", (30.6, 18.2, 14.0), "motion x600"),
    ("Anker Soundcore", "Flare 2", "s", 75, 88, "360", (9.0, 16.1, 9.0), "flare 2"),
    ("Anker Soundcore", "Boom 2", "m", 50, 94, "frontal", (24.5, 9.8, 9.4), "soundcore boom 2"),
    ("Anker Soundcore", "Rave Neo 2", "xl", 45, 100, "frontal", (25.5, 33.0, 23.0), "rave neo"),
    ("Marshall", "Willen", "xs", 110, 82, "frontal", (10.1, 10.0, 4.0), ""),
    ("Marshall", "Willen II", "xs", 100, 83, "frontal", (10.1, 10.0, 4.0), ""),
    ("Marshall", "Emberton", "s", 70, 87, "360", (16.0, 6.8, 7.6), ""),
    ("Marshall", "Emberton II", "s", 65, 88, "360", (16.0, 6.8, 7.6), ""),
    ("Marshall", "Emberton III", "s", 60, 89, "360", (16.0, 7.6, 6.8), ""),
    ("Marshall", "Middleton", "m", 50, 93, "360", (23.0, 10.9, 9.9), ""),
    ("Marshall", "Kilburn II", "m", 50, 94, "360", (24.3, 16.2, 14.0), ""),
    ("Marshall", "Tufton", "l", 40, 99, "360", (22.9, 35.0, 16.3), ""),
    ("Marshall", "Acton II", "m", 45, 96, "estereo", (26.5, 18.8, 15.1), ""),
    ("Marshall", "Acton III", "m", 45, 96, "estereo", (26.5, 18.5, 15.3), ""),
    ("Marshall", "Stanmore II", "l", 40, 99, "estereo", (35.0, 19.5, 18.5), ""),
    ("Marshall", "Stanmore III", "l", 40, 100, "estereo", (35.0, 20.3, 18.8), ""),
    ("Marshall", "Woburn II", "xl", 35, 104, "estereo", (40.0, 31.7, 20.0), ""),
    ("Beats", "Pill+", "s", 70, 88, "estereo", (21.2, 6.2, 6.2), "beats pill plus"),
    ("Beats", "Pill (2024)", "s", 60, 90, "frontal", (21.0, 6.6, 7.0), "beats pill"),
    ("Bang & Olufsen", "Beosound A1 2nd Gen", "s", 60, 88, "360", (13.3, 4.6, 13.3), "beosound a1"),
    ("Bang & Olufsen", "Beosound Explore", "s", 60, 89, "360", (8.1, 12.4, 8.1), "beosound explore"),
    ("Bang & Olufsen", "Beosound A5", "l", 40, 98, "360", (30.0, 22.0, 18.0), "beosound a5"),
    ("Bang & Olufsen", "Beoplay P6", "s", 60, 89, "360", (16.5, 17.5, 7.0), "beoplay p6"),
    ("Harman Kardon", "Onyx Studio 7", "l", 50, 95, "estereo", (28.6, 25.3, 13.0), "onyx studio 7"),
    ("Harman Kardon", "Onyx Studio 8", "l", 50, 96, "estereo", (28.8, 26.9, 13.6), "onyx studio 8"),
    ("Harman Kardon", "Go + Play 3", "l", 45, 98, "estereo", (41.6, 25.6, 18.4), "go play"),
    ("Harman Kardon", "Luna", "s", 60, 89, "frontal", (19.8, 8.4, 8.0), "harman luna"),
    ("Harman Kardon", "Aura Studio 4", "l", 45, 96, "360", (20.8, 32.7, 20.8), "aura studio"),
    ("Tribit", "XSound Go", "s", 75, 88, "frontal", (17.5, 6.6, 5.9), "xsound go"),
    ("Tribit", "XSound Plus 2", "s", 65, 89, "estereo", (19.0, 7.0, 6.5), "xsound plus"),
    ("Tribit", "StormBox Micro 2", "xs", 90, 85, "frontal", (11.1, 10.7, 4.0), "stormbox micro"),
    ("Tribit", "StormBox", "s", 65, 89, "360", (7.4, 17.5, 7.4), "stormbox 360"),
    ("Tribit", "StormBox 2", "m", 55, 92, "estereo", (20.6, 8.8, 8.3), "stormbox 2"),
    ("Tribit", "StormBox Blast", "l", 45, 98, "estereo", (39.0, 19.4, 15.4), "stormbox blast"),
    ("Sonos", "Roam", "s", 65, 88, "frontal", (6.2, 16.8, 6.0), "sonos roam"),
    ("Sonos", "Roam 2", "s", 62, 88, "frontal", (6.2, 16.8, 6.0), "sonos roam 2"),
    ("Sonos", "Move", "l", 45, 95, "frontal", (16.0, 24.1, 12.7), "sonos move"),
    ("Sonos", "Move 2", "l", 40, 97, "estereo", (16.0, 24.1, 12.7), "sonos move 2"),
    ("LG", "XBOOM Go PL2", "xs", 110, 83, "frontal", (9.2, 9.0, 8.4), "pl2"),
    ("LG", "XBOOM Go PL5", "s", 70, 88, "frontal", (22.0, 8.5, 8.0), "pl5"),
    ("LG", "XBOOM Go PL7", "m", 55, 92, "frontal", (25.5, 10.0, 10.4), "pl7"),
    ("LG", "XBOOM Grab", "s", 65, 89, "frontal", (20.0, 8.0, 7.5), "xboom grab"),
    ("Samsung", "Sound Tower MX-ST40B", "xl", 45, 100, "frontal", (28.1, 54.2, 28.8), "sound tower"),
    ("Xiaomi", "Mi Portable Speaker 16W", "s", 75, 87, "frontal", (21.4, 7.2, 7.2), "mi portable speaker"),
    ("Xiaomi", "Sound Pocket", "xs", 110, 82, "frontal", (8.0, 10.6, 4.0), "sound pocket"),
    ("Amazon", "Echo Dot (BT)", "xs", 100, 84, "360", (10.0, 9.0, 10.0), "echo dot"),
    ("Amazon", "Echo (4.ª gen, BT)", "m", 50, 92, "360", (14.4, 13.3, 14.4), "echo"),
    ("Apple", "Beats Pill (2015)", "s", 75, 87, "estereo", (21.2, 6.2, 6.2), ""),
    ("Philips", "TAS2505", "xs", 110, 82, "frontal", (9.5, 9.5, 4.5), ""),
    ("Philips", "TAS5505", "s", 70, 88, "frontal", (20.0, 8.0, 8.0), ""),
    ("Edifier", "MP230", "s", 70, 87, "frontal", (19.0, 13.0, 8.0), ""),
    ("Edifier", "MP260", "s", 70, 87, "frontal", (19.0, 13.0, 8.0), ""),
    ("Edifier", "R1280DB", "l", 55, 95, "frontal", (14.6, 23.4, 19.6), "edifier r1280"),
    ("Edifier", "R1700BT", "l", 50, 96, "frontal", (15.6, 25.0, 22.5), "edifier r1700"),
    ("Edifier", "MR4 (BT)", "m", 60, 92, "frontal", (14.5, 22.0, 17.5), ""),
    ("Audioengine", "A2+ Wireless", "m", 65, 93, "frontal", (10.2, 15.2, 13.3), "a2+"),
    ("Audioengine", "A5+ Wireless", "l", 50, 99, "frontal", (17.8, 27.4, 19.0), "a5+"),
    ("Klipsch", "The One Plus", "l", 45, 97, "estereo", (34.4, 17.3, 13.7), "klipsch the one"),
    ("Klipsch", "The Three Plus", "l", 40, 99, "estereo", (34.9, 21.6, 18.4), "klipsch the three"),
    ("Klipsch", "R-41PM", "l", 55, 100, "frontal", (17.1, 27.6, 21.3), "klipsch r-41pm"),
    ("Creative", "Pebble", "s", 100, 84, "frontal", (11.6, 12.2, 11.5), "creative pebble"),
    ("Creative", "Stage SE", "m", 60, 90, "estereo", (55.0, 7.1, 9.5), ""),
    ("Generic", "Parlante de bolsillo", "xs", 150, 80, "frontal", (8, 8, 4), "generico xs"),
    ("Generic", "Parlante pequeño", "s", 120, 85, "frontal", (15, 8, 6), "generico s"),
    ("Generic", "Parlante mediano", "m", 70, 90, "frontal", (22, 10, 9), "generico m"),
    ("Generic", "Parlante grande", "l", 55, 96, "frontal", (30, 15, 13), "generico l"),
    ("Generic", "Torre / fiesta", "xl", 45, 102, "frontal", (30, 55, 30), "generico xl"),
    ("Generic", "Parlantes de la Mac", "xs", 120, 82, "estereo", (30, 1, 20), "bocinas de macbook, macbook speakers"),
]

LIBRARY = [
    {"id": f"{b}|{m}".lower(), "brand": b, "model": m, "size": sz, "low_hz": lo, "spl": spl, "disp": disp,
     "dims": dims, "aliases": [a.strip() for a in al.split(",") if a.strip()]}
    for b, m, sz, lo, spl, disp, dims, al in _RAW
]
BY_ID = {e["id"]: e for e in LIBRARY}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", s.lower()).strip()


def match(device_name: str) -> dict | None:
    """El modelo más probable según el nombre Bluetooth (p. ej. «JBL Clip 4» → JBL Clip 4)."""
    n = _norm(device_name)
    best, score = None, 0
    for e in LIBRARY:
        keys = [_norm(f"{e['brand']} {e['model']}"), _norm(e["model"])] + [_norm(a) for a in e["aliases"]]
        for k in keys:
            if not k:
                continue
            toks = k.split()
            hit = sum(1 for t in toks if t in n.split())
            if k in n:
                s = 100 + len(k)
            elif toks and hit == len(toks):
                s = 50 + len(k)
            elif hit >= 2:
                s = 10 * hit
            else:
                continue
            if s > score:
                best, score = e, s
    return best if score >= 20 else None


def search(q: str, limit: int = 30) -> list:
    n = _norm(q)
    if not n:
        return LIBRARY[:limit]
    out = [e for e in LIBRARY if all(t in _norm(f"{e['brand']} {e['model']} {' '.join(e['aliases'])}") for t in n.split())]
    return out[:limit]
