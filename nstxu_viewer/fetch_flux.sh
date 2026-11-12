#!/bin/bash
# Everything about reaching PPPL Flux for the viewer. Copies NSTX-U shots' measured data from Flux into
# runs/flux/<shot>.h5 (~10 MB a shot): Thomson (MPTS), EFIT01 and CHERS as the sim-init workflow's slim files, plus
# the beam power WF:\PNB (the flattop). The extraction is the workflow's tools/extract_slim_h5.py, run on Flux in a
# temporary directory removed when it ends.
#
# Perlmutter has no route to PPPL; the laptop has one. Two ways:
#  ON PERLMUTTER, through the laptop: the laptop's ssh connection to Perlmutter forwards Flux's ssh port to a socket
#    on the login node it lands on (laptop ~/.ssh/config, in the Perlmutter host entry:
#        RemoteForward /tmp/<your Perlmutter user name>-flux.sock flux.pppl.gov:22
#    then reconnect). On that login node:
#        ./fetch_flux.sh 203655 203899      asks for the Flux password + Duo
#        ./view.sh --flux                   asks once at start-up; the viewer's Fetch button then uses that login
#  ON THE LAPTOP:  NSTXU_VIEWER_DATA=<the data directory on Perlmutter> bash <(ssh perlmutter cat <this file>) 203655
#    (FLUX, PM: the laptop's ssh names for Flux and Perlmutter)
# view.sh's: fetch_flux.sh --login | --check | --logout <control path>  (a master connection kept for the viewer)
# Settings: the data directory's flux.env (shell assignments, see flux.env.example; read over ssh on the laptop):
# FLUX_HOST (Flux's user@host), FLUX_PY (a Flux python with h5py), FLUX_TOOL (the sim-init workflow's
# tools/extract_slim_h5.py on Perlmutter). FLUX_SOCK: the forwarded socket; FLUX_DEST: where the files go
# (default: the data directory's runs/flux). The data directory: $NSTXU_VIEWER_DATA (view.sh exports it), else here.
set -euo pipefail
DATA=${NSTXU_VIEWER_DATA:-$(cd "$(dirname "$0")" && pwd)}
if [ -r "$DATA/flux.env" ]; then
    . "$DATA/flux.env"
elif ENV=$(ssh -x "${PM:-perlmutter}" cat "$DATA/flux.env"); then   # on the laptop
    eval "$ENV"
else
    echo "no $DATA/flux.env here or on ${PM:-perlmutter}: set NSTXU_VIEWER_DATA to the viewer's data directory" >&2
    exit 2
fi
: "${FLUX_HOST:?set in flux.env}" "${FLUX_PY:?set in flux.env}" "${FLUX_TOOL:?set in flux.env}"
TOOL=$FLUX_TOOL
DEST=${FLUX_DEST:-$DATA/runs/flux}
FLUX_SOCK=${FLUX_SOCK:-/tmp/$USER-flux.sock}
# -x: Perlmutter's ssh_config forwards X11 to every host, and Flux then warns that xauth failed
VIA_SOCK=(-x -o "ProxyCommand=socat - UNIX-CONNECT:$FLUX_SOCK" -o ServerAliveInterval=60 -o ServerAliveCountMax=3)
NO_SOCK="no $FLUX_SOCK on $(hostname): the laptop forwards it only to the login node its connection lands on
(laptop ~/.ssh/config, Perlmutter host entry: RemoteForward $FLUX_SOCK flux.pppl.gov:22, then reconnect)"

case "${1:-}" in
    --login)   # asks for the password + Duo here, then serves commands at control path $2 until --logout
        [ -S "$FLUX_SOCK" ] || { echo "$NO_SOCK" >&2; exit 4; }
        exec ssh "${VIA_SOCK[@]}" -M -S "$2" -f -N "$FLUX_HOST" ;;
    --check)   exec ssh -S "$2" -O check "$FLUX_HOST" 2>/dev/null ;;
    --logout)  exec ssh -S "$2" -O exit "$FLUX_HOST" 2>/dev/null ;;
esac

[ $# -ge 1 ] || { echo "usage: fetch_flux.sh <shot> [<shot> ...]   (see the top of this file)" >&2; exit 2; }
for s in "$@"; do [[ $s =~ ^[0-9]{6}$ ]] || { echo "not a shot number: $s" >&2; exit 2; }; done
SHOTS=$(IFS=,; echo "$*")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
if [ -r "$TOOL" ]; then   # on Perlmutter
    if [ -n "${FLUX_CM:-}" ] && ssh -S "$FLUX_CM" -O check "$FLUX_HOST" 2>/dev/null; then
        FLUX_SSH=(ssh -x -S "$FLUX_CM" -o BatchMode=yes "$FLUX_HOST")   # view.sh --flux's login
    elif [ -S "$FLUX_SOCK" ]; then
        FLUX_SSH=(ssh "${VIA_SOCK[@]}" "$FLUX_HOST")
    else
        echo "$NO_SOCK" >&2; exit 4
    fi
    cp "$TOOL" "$WORK/extract_slim_h5.py"
    deliver() { mkdir -p "$DEST" && tar -C "$DEST" -xf - && ls -l "$DEST"; }
else                      # on the laptop
    PM=${PM:-perlmutter}
    FLUX_SSH=(ssh -x "${FLUX:-flux}")
    echo "== reading the extraction tool from $PM"
    ssh -x "$PM" cat "$TOOL" > "$WORK/extract_slim_h5.py"
    deliver() { ssh -x "$PM" "mkdir -p '$DEST' && tar -C '$DEST' -xf - && ls -l '$DEST'"; }
fi

echo "== extracting $SHOTS on Flux (its log below)"
# Flux's login shell may not be bash: the remote script goes in on stdin to 'bash -s', the tool inside it as a
# heredoc. Flux's login prints its banner and module lines, maybe to stdout: the files come back base64-encoded
# between two marker lines
{
    printf 'PY=%q\nSHOTS=%q\n' "$FLUX_PY" "$SHOTS"
    cat <<'EOF'
set -euo pipefail
[ -x "$PY" ] || PY=python3
"$PY" -c 'import h5py' 1>&2 || { echo "no h5py in $PY: set FLUX_PY to a Flux python with h5py" >&2; exit 3; }
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cat > "$d/extract_slim_h5.py" <<'PYEOF'
EOF
    cat "$WORK/extract_slim_h5.py"
    cat <<'EOF'
PYEOF
"$PY" "$d/extract_slim_h5.py" "$SHOTS" --out "$d/out" --extra-node 'WF=\WF::TOP:PNB' 1>&2
echo @@FLUX-SLIM-BEGIN@@
tar -C "$d/out" -cf - . | base64
echo @@FLUX-SLIM-END@@
EOF
} | "${FLUX_SSH[@]}" bash -s > "$WORK/flux.out"
awk '/^@@FLUX-SLIM-END@@$/ {f = 0} f; /^@@FLUX-SLIM-BEGIN@@$/ {f = 1}' "$WORK/flux.out" | base64 --decode > "$WORK/slim.tar"
mkdir "$WORK/out"
tar -C "$WORK/out" -xf "$WORK/slim.tar"
for s in "$@"; do
    [ -s "$WORK/out/$s.h5" ] || { echo "Flux sent no $s.h5; its output began:" >&2; head -5 "$WORK/flux.out" >&2; exit 5; }
done
echo "== delivering to $DEST"
tar -C "$WORK/out" -cf - . | deliver
echo "done: in the viewer, press 'reload frames' (or reopen the shot)"
