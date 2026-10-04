# Surround BT

Sonido envolvente con los parlantes Bluetooth que ya tienes. Conecta 2 o 3 parlantes a la Mac, ubícalos en el plano de tu cuarto y cada uno recibe su propio canal. El micrófono de la Mac los sincroniza y les iguala el volumen y el tono.

## Descargar

**[Bajar Surround BT para Mac (.dmg)](https://github.com/danielmatzenj-png/surround-bt/releases/latest/download/Surround-BT.dmg)**

## Requisitos

- Mac con chip Apple (M1 o posterior) y macOS 12 o posterior. En Macs Intel no funciona.
- 2 o 3 parlantes Bluetooth. Con más, el Bluetooth de la Mac no da abasto. Algunos parlantes, como el Echo Dot, se llevan mal con los demás.
- Para que suene el audio de la Mac (Netflix, YouTube, Spotify): [BlackHole 2ch](https://existential.audio/blackhole/), que es gratis. La app te explica cómo configurarlo.

## Instalar

1. Abre el `.dmg` y arrastra **Surround BT** a Aplicaciones.
2. Ábrela. Como no está firmada por Apple, macOS la bloquea: toca **Listo**.
3. Ve a **Ajustes del Sistema → Privacidad y seguridad**, baja hasta el final y toca **Abrir igualmente**.
4. Acepta los permisos de **micrófono** (para calibrar) y **Bluetooth** (para reconectar los parlantes).

Solo hace falta la primera vez.

## Primeros pasos

1. Conecta los parlantes por Bluetooth desde el menú de la Mac.
2. En la app, agrégalos y ubícalos en el plano según dónde están en tu cuarto.
3. Pon la Mac donde te sientas y toca **Calibrar**. Mide cada parlante con el micrófono e iguala tiempos, volumen y tono.

## Componentes de terceros

La app incluye [three.js](https://threejs.org) (MIT), [PortAudio](https://www.portaudio.com) (MIT), [NumPy](https://numpy.org) y [SciPy](https://scipy.org) (BSD) y [FFmpeg](https://ffmpeg.org) (LGPL/GPL, vía imageio-ffmpeg; código fuente en ffmpeg.org).
