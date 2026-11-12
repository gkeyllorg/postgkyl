#!/bin/bash
# Serve the NSTX-U run viewer and open it in the browser.
#   ./view.sh [run|edit] [--KEY VALUE ...]
#   ./view.sh                                    app view (controls in the sidebar, code hidden)
#   ./view.sh edit                               notebook view (every cell visible and editable)
#   ./view.sh --shot 203899 --frame 30 --qty T_elc        open on a shot, a frame and a quantity
#   ./view.sh --sim 'Sim 1' --run prod_D1_coll --frames 0:60 --probe 10,0,15 --tab Probe
#   ./view.sh --state runs/states/my_view.json   a view saved from the page (Save & export)
#   ./view.sh --path DIR --blocks 1-3 --cmp 'Sim 1/prod_coll'   any run directory, some blocks, another run over it
#   ./view.sh --flux                             log in to PPPL Flux first (password + Duo): Fetch from Flux works
#   ./view.sh --help                             every key
#   OPEN=0 ./view.sh      do not open a browser: print the URL only
#   PORT=2750 ./view.sh   a fixed port (default: marimo picks a free one)
#   NSTXU_VIEWER_DATA=DIR   the data directory: viewer.json (the simulations), flux.env, runs/ (default: here)
#   NSTXU_VIEWER_VENV=DIR   the Python environment with the viewer's packages, requirements.lock.txt (default: ./.venv)
# In a VS Code terminal, $BROWSER is VS Code's opener: the page opens on your own machine through the port VS Code
# forwards, in the browser named by the VS Code setting workbench.externalBrowser ("chrome" in this workspace's
# .vscode/settings.json). Elsewhere: forward the port with ssh -L and open the printed URL in Chrome.
# The access token stays on: anyone on a shared login node can reach 127.0.0.1. Ctrl-C stops the server.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export NSTXU_VIEWER_DATA=$(cd "${NSTXU_VIEWER_DATA:-$HERE}" && pwd)
BIN=${NSTXU_VIEWER_VENV:-$HERE/.venv}/bin
[ -x "$BIN/marimo" ] || { echo "no marimo in $BIN: set NSTXU_VIEWER_VENV (see README.md)" >&2; exit 2; }
MODE=run
case "${1:-}" in run|edit) MODE=$1; shift ;; esac
if [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    echo; echo "Keys (also the page's address, and a saved view):"
    cd "$HERE" && "$BIN/python" -c 'import viewer_state as vs
for k, (what, eg) in vs.KEYS.items(): print(f"  --{k:<7} {what}   e.g. --{k} {eg}")'
    exit 0
fi
# the view keys go to the notebook (mo.cli_args); a state file's path is made absolute before the cd below
ARGS=()
FLUX=0
while [ $# -gt 0 ]; do
    case "$1" in
        --flux)  FLUX=1; shift ;;
        --state) [ $# -ge 2 ] || { echo "--state needs a file" >&2; exit 2; }
                 [ -f "$2" ] || { echo "no state file $2" >&2; exit 2; }
                 ARGS+=(--state "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2 ;;
        --path|--cmp)   # a run directory: absolute before the cd below (a '<simulation>/<run>' cmp stays as it is)
                 [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
                 if [ -d "$2" ]; then ARGS+=("$1" "$(cd "$2" && pwd)"); else ARGS+=("$1" "$2"); fi; shift 2 ;;
        --*)     [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
                 ARGS+=("$1" "$2"); shift 2 ;;
        *)       echo "unexpected argument $1 (see ./view.sh --help)" >&2; exit 2 ;;
    esac
done
cd "$HERE"
mkdir -p runs "$NSTXU_VIEWER_DATA/runs"
SERVER= LOG= FLUX_DIR=
cleanup() {
    [ -n "$SERVER" ] && kill $SERVER 2>/dev/null
    [ -n "$LOG" ] && rm -f "$LOG"
    if [ -n "$FLUX_DIR" ]; then bash "$HERE/fetch_flux.sh" --logout "$FLUX_DIR/cm" || true; rm -rf "$FLUX_DIR"; fi
}
trap cleanup EXIT
if [ "$FLUX" = 1 ]; then
    # one Flux login (fetch_flux.sh: through the socket the laptop forwards), kept for the viewer's Fetch button and
    # closed when the viewer stops; its control socket in a directory only you can open
    FLUX_DIR=$(mktemp -d /tmp/flux-cm.XXXXXX)
    echo "Flux login (password, then a Duo passcode, or 1 for a push):"
    if bash fetch_flux.sh --login "$FLUX_DIR/cm"; then
        export FLUX_CM=$FLUX_DIR/cm
    else
        echo "no Flux login: the viewer starts without it" >&2
        rm -rf "$FLUX_DIR"; FLUX_DIR=
    fi
fi
# the page's font and dark-mode surfaces, from the poster theme (nstxu_viewer.py loads runs/viewer.css, next to it)
"$BIN/python" -c 'import viewer_plots as vp; open("runs/viewer.css", "w").write(vp.page_css())'
LOG=$NSTXU_VIEWER_DATA/runs/view_$$.log
# -y: Ctrl-C stops marimo at once, without asking
"$BIN/marimo" -y "$MODE" nstxu_viewer.py --headless --token ${PORT:+--port $PORT} -- "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER=$!
tail -n +1 -f "$LOG" --pid=$SERVER &
URL=
for _ in $(seq 600); do   # up to 60 s for the server to print its URL
    URL=$(grep -o 'http://[^ ]*access_token=[A-Za-z0-9_-]*' "$LOG" | head -1 || true)
    [ -n "$URL" ] && break
    kill -0 $SERVER 2>/dev/null || { echo "marimo exited" >&2; exit 1; }
    sleep 0.1
done
[ -n "$URL" ] || { echo "no URL from marimo after 60 s" >&2; exit 1; }
if [ "${OPEN:-1}" = 1 ] && [ -n "${BROWSER:-}" ]; then
    "$BROWSER" "$URL" >/dev/null 2>&1 || echo "could not open a browser: open $URL" >&2
elif [ "${OPEN:-1}" = 1 ]; then
    PORTNUM=$(echo "$URL" | sed -E 's|http://[^:]+:([0-9]+).*|\1|')
    echo "No \$BROWSER here (not a VS Code terminal). On your machine:"
    echo "  ssh -L $PORTNUM:localhost:$PORTNUM $(hostname)    then open in Chrome:  $URL"
fi
wait $SERVER
