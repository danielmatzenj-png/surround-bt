#!/bin/zsh
# Doble clic: arranca Surround BT y abre la app en el navegador.
cd "$(dirname "$0")"
if ! curl -s -o /dev/null http://localhost:8890/; then
  python3 -m sbt ui > /tmp/surround-bt.log 2>&1 &
  for i in {1..20}; do curl -s -o /dev/null http://localhost:8890/ && break; sleep 0.3; done
fi
open http://localhost:8890/
