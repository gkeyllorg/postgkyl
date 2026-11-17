#!/bin/bash
# Everything about reaching PPPL Flux for the viewer. Copies NSTX-U shots' measured data from Flux into
# runs/flux/<shot>.h5 (~10 MB a shot): Thomson (MPTS), EFIT01 and CHERS as the sim-init workflow's slim files, plus
# the beam power WF:\PNB (the flattop), plus one TRANSP run's T_i, n_D and midplane radii (/<shot>/TRANSP: which
# run and why, by flux_transp.py's rule). The extraction is the workflow's tools/extract_slim_h5.py and this viewer's
# flux_transp.py, run on Flux in a temporary directory removed when it ends.
#
# Perlmutter has no route to PPPL; the laptop has one. Two ways:
#  ON PERLMUTTER, through the laptop: the laptop's ssh connection to Perlmutter forwards Flux's ssh port to a socket
#    on the login node it lands on (laptop ~/.ssh/config, in the Perlmutter host entry:
#        RemoteForward /tmp/<your Perlmutter user name>-flux.sock flux.pppl.gov:22
#    then reconnect). On that login node:
#        ./fetch_flux.sh 203655 203899      asks for the Flux password + Duo
#        ./view.sh --flux                   asks once at start-up; the viewer's Fetch button then uses that login
#  ON THE LAPTOP:  GK_VIEWER_DATA=<the data directory on Perlmutter> bash <(ssh perlmutter cat <this file>) 203655
#    (FLUX, PM: the laptop's ssh names for Flux and Perlmutter)
# view.sh's: fetch_flux.sh --login | --check | --logout <control path>  (a master connection kept for the viewer)
# Settings: the data directory's flux.env (shell assignments, see flux.env.example; read over ssh on the laptop):
# FLUX_HOST (Flux's user@host), FLUX_PY (a Flux python with h5py), FLUX_TOOL (the sim-init workflow's
# tools/extract_slim_h5.py on Perlmutter). FLUX_SOCK: the forwarded socket; FLUX_DEST: where the files go
# (default: the data directory's runs/flux). The data directory: $GK_VIEWER_DATA (view.sh exports it), else here.
# Optional: FLUX_TRANSP_CSV (the sim-init workflow's candidate CSV on Perlmutter, whose TRANSP_RUN_ID is the run taken
# for a shot it lists; default: FLUX_TOOL's workflow's code/elong_tri_scan_pre_shape_candidates_latest.csv; empty:
# none), FLUX_TRANSP_TOOL (this viewer's flux_transp.py on Perlmutter; needed on the laptop only), FLUX_TRANSP_ROOT
# (TRANSP's archive on Flux, default /p/transparch/result/NSTU), FLUX_H5_ROOT (the MDSplus dumps' directory on Flux
# instead of the extraction tool's two: a stand-in Flux).
set -euo pipefail
DATA=${GK_VIEWER_DATA:-$(cd "$(dirname "$0")" && pwd)}
if [ -r "$DATA/flux.env" ]; then
    . "$DATA/flux.env"
elif ENV=$(ssh -x "${PM:-perlmutter}" cat "$DATA/flux.env"); then   # on the laptop
    eval "$ENV"
else
    echo "no $DATA/flux.env here or on ${PM:-perlmutter}: set GK_VIEWER_DATA to the viewer's data directory" >&2
    exit 2
fi
: "${FLUX_HOST:?set in flux.env}" "${FLUX_PY:?set in flux.env}" "${FLUX_TOOL:?set in flux.env}"
TOOL=$FLUX_TOOL
TRANSP_TOOL=${FLUX_TRANSP_TOOL:-$(cd "$(dirname "$0")" && pwd)/flux_transp.py}
PICKS=${FLUX_TRANSP_CSV-$(dirname "$(dirname "$TOOL")")/code/elong_tri_scan_pre_shape_candidates_latest.csv}
DEST=${FLUX_DEST:-$DATA/runs/flux}
FLUX_SOCK=${FLUX_SOCK:-/tmp/$USER-flux.sock}
# -x: Perlmutter's ssh_config forwards X11 to every host, and Flux then warns that xauth failed
VIA_SOCK=(-x -o "ProxyCommand=socat - UNIX-CONNECT:$FLUX_SOCK" -o ServerAliveInterval=60 -o ServerAliveCountMax=3)
RECONNECT="A NEW ssh connection from the laptop forwards it, to the login node that connection lands on (a VS Code window
reload keeps the old connection: close the remote connection, or open a plain ssh from a laptop terminal)."
NO_SOCK="no $FLUX_SOCK on $(hostname).
$RECONNECT
(laptop ~/.ssh/config, Perlmutter host entry: RemoteForward $FLUX_SOCK flux.pppl.gov:22)"

# The laptop's forward is there and answers. sshd does not remove the socket when the laptop disconnects, and a later
# connection landing on this node cannot make it again while the file is there ("Connection refused" from socat):
# a dead one of ours is removed here so that the next connection can
sock_ready() {
    [ -S "$FLUX_SOCK" ] || { echo "$NO_SOCK" >&2; return 1; }
    timeout 10 socat -u /dev/null "UNIX-CONNECT:$FLUX_SOCK" 2>/dev/null && return 0
    local made
    made=$(date -r "$FLUX_SOCK" '+%F %R')
    if [ -O "$FLUX_SOCK" ] && rm -f "$FLUX_SOCK"; then
        echo "$FLUX_SOCK on $(hostname) was left by a laptop connection that has ended (made $made): removed it." >&2
    else
        echo "$FLUX_SOCK on $(hostname) was left by a laptop connection that has ended (made $made): rm it." >&2
    fi
    echo "$RECONNECT" >&2
    return 1
}

case "${1:-}" in
    --login)   # asks for the password + Duo here, then serves commands at control path $2 until --logout
        sock_ready || exit 4
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
    elif sock_ready; then
        FLUX_SSH=(ssh "${VIA_SOCK[@]}" "$FLUX_HOST")
    else
        exit 4
    fi
    cp "$TOOL" "$WORK/extract_slim_h5.py"
    cp "$TRANSP_TOOL" "$WORK/flux_transp.py"
    if [ -n "$PICKS" ] && [ -r "$PICKS" ]; then cp "$PICKS" "$WORK/picks.csv"; else : > "$WORK/picks.csv"; fi
    deliver() { mkdir -p "$DEST" && tar -C "$DEST" -xf - && ls -l "$DEST"; }
else                      # on the laptop
    PM=${PM:-perlmutter}
    FLUX_SSH=(ssh -x "${FLUX:-flux}")
    echo "== reading the extraction tools from $PM"
    ssh -x "$PM" cat "$TOOL" > "$WORK/extract_slim_h5.py"
    ssh -x "$PM" cat "$TRANSP_TOOL" > "$WORK/flux_transp.py" ||
        { echo "set FLUX_TRANSP_TOOL to this viewer's flux_transp.py on $PM" >&2; exit 2; }
    { [ -n "$PICKS" ] && ssh -x "$PM" cat "$PICKS"; } > "$WORK/picks.csv" 2>/dev/null || : > "$WORK/picks.csv"
    deliver() { ssh -x "$PM" "mkdir -p '$DEST' && tar -C '$DEST' -xf - && ls -l '$DEST'"; }
fi

if [ -s "$WORK/picks.csv" ]; then
    echo "== TRANSP: the sim-init workflow's run where $PICKS lists the shot, else flux_transp.py's rule"
else
    echo "== TRANSP: flux_transp.py's rule (no candidate CSV${PICKS:+ at $PICKS})"
fi
echo "== extracting $SHOTS on Flux (its log below)"
# Flux's login shell may not be bash: the remote script goes in on stdin to 'bash -s', the tools and the CSV inside
# it as heredocs. Flux's login prints its banner and module lines, maybe to stdout: the files come back
# base64-encoded between two marker lines
{
    printf 'PY=%q\nSHOTS=%q\nTRANSP_ROOT=%q\nH5_ROOT=%q\n' "$FLUX_PY" "$SHOTS" \
        "${FLUX_TRANSP_ROOT:-/p/transparch/result/NSTU}" "${FLUX_H5_ROOT:-}"
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
cat > "$d/flux_transp.py" <<'PYEOF'
EOF
    cat "$WORK/flux_transp.py"
    cat <<'EOF'
PYEOF
cat > "$d/picks.csv" <<'CSVEOF'
EOF
    cat "$WORK/picks.csv"
    echo   # the CSV may end without a newline: the marker below must start a line
    cat <<'EOF'
CSVEOF
"$PY" "$d/extract_slim_h5.py" "$SHOTS" --out "$d/out" --extra-node 'WF=\WF::TOP:PNB' \
    ${H5_ROOT:+--source-root "$H5_ROOT"} 1>&2
"$PY" "$d/flux_transp.py" "$SHOTS" --out "$d/out" --root "$TRANSP_ROOT" --picks "$d/picks.csv" 1>&2
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
