#!/bin/zsh
# Arma «Surround BT.app» y «Surround BT.dmg» en packaging/out/. Uso: ./packaging/build.sh
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$ROOT/packaging/out
rm -rf "$OUT" "$ROOT/packaging/build/dist" "$ROOT/packaging/build/work" "$ROOT/packaging/build/dmg"
mkdir -p "$OUT"

# Entorno limpio: solo lo que la app usa (si no, PyInstaller arrastra todo lo instalado en la Mac).
VENV="$ROOT/packaging/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "0/5 Entorno limpio…"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q numpy scipy sounddevice soundfile imageio-ffmpeg "qrcode[pil]" pyinstaller
fi
PY="$VENV/bin/python"

# Herramienta de reconexión Bluetooth (no se versiona el binario)
swiftc -O -target arm64-apple-macos12.0 "$ROOT/bin/btconnect.swift" -o "$ROOT/bin/btconnect" -framework IOBluetooth

echo "1/5 Motor (PyInstaller)…"
"$PY" -m PyInstaller --noconfirm --log-level WARN --onedir --name sbt-server \
  --distpath "$ROOT/packaging/build/dist" --workpath "$ROOT/packaging/build/work" --specpath "$ROOT/packaging/build" \
  --paths "$ROOT" \
  --add-data "$ROOT/ui:ui" --add-binary "$ROOT/bin/btconnect:bin" \
  --collect-all imageio_ffmpeg --collect-all _sounddevice_data --collect-submodules sbt \
  --hidden-import qrcode --hidden-import qrcode.image.pil --hidden-import PIL.Image \
  --hidden-import scipy.signal --hidden-import scipy.special._cdflib \
  --exclude-module tkinter --exclude-module matplotlib --exclude-module IPython \
  "$ROOT/packaging/server_main.py"

echo "2/5 App nativa (Swift)…"
APP="$OUT/Surround BT.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -target arm64-apple-macos12.0 "$ROOT/packaging/SurroundBT.swift" -o "$APP/Contents/MacOS/SurroundBT"
cp "$ROOT/packaging/Info.plist" "$APP/Contents/Info.plist"
cp -R "$ROOT/packaging/build/dist/sbt-server" "$APP/Contents/Resources/server"

echo "3/5 Ícono…"
ICONSET="$ROOT/packaging/build/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$ROOT/ui/icon-512.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  d=$((s*2)); sips -z $d $d "$ROOT/ui/icon-512.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

echo "4/5 Firma local (ad hoc)…"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1

echo "5/5 Imagen de disco…"
STAGE="$ROOT/packaging/build/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Aplicaciones"
cp "$ROOT/packaging/LEEME.txt" "$STAGE/LÉEME.txt"
hdiutil create -quiet -volname "Surround BT" -srcfolder "$STAGE" -ov -format UDZO "$OUT/Surround BT.dmg"
du -sh "$APP" "$OUT/Surround BT.dmg"
echo "Listo: $OUT"
