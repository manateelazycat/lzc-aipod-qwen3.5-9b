#!/bin/sh
set -eu

MANIFEST="${QWEN35_9B_CONTROL_MANIFEST:-/lzcapp/pkg/content/control-manifest.json}"
DATA_ROOT="${QWEN35_9B_DATA_DIR:-/lzcapp/var/qwen35-9b-defiant-fable}"
CACHE="$DATA_ROOT/control-plane"

# Explicit development-only override for offline unit/smoke verification of bootstrap.sh.
if [ -n "${QWEN35_9B_DEV_CONTROL_BIN:-}" ]; then
  if [ ! -x "$QWEN35_9B_DEV_CONTROL_BIN" ]; then
    echo "Development controller binary not executable: $QWEN35_9B_DEV_CONTROL_BIN" >&2
    exit 1
  fi
  exec "$QWEN35_9B_DEV_CONTROL_BIN"
fi

if [ ! -f "$MANIFEST" ]; then
  echo "Control manifest missing: $MANIFEST" >&2
  exit 1
fi

read_json_field() {
  key="$1"
  awk -F'"' -v k="$key" '
    $0 ~ "\"" k "\"[[:space:]]*:" {
      for (i = 1; i <= NF; i++) {
        if ($i == k) {
          val = $(i + 2)
          if (val == "") {
            line = $0
            sub(/^[^:]*:[[:space:]]*/, "", line)
            sub(/[[:space:]]*,?[[:space:]]*$/, "", line)
            print line
          } else {
            print val
          }
          exit
        }
      }
    }
  ' "$MANIFEST"
}

BUNDLE_SHA="$(read_json_field sha256)"
BUNDLE_SIZE="$(read_json_field size)"
BIN_SHA="$(read_json_field binarySha256)"
MS_URL="$(read_json_field modelscopeUrl)"
HF_URL="$(read_json_field huggingfaceUrl)"
BIN_NAME="$(read_json_field binaryName)"
[ -n "$BIN_NAME" ] || BIN_NAME="qwen35-9b-lpk"

case "$BUNDLE_SHA" in
  ''|*'PENDING'*|*[!0-9a-fA-F]*)
    echo "Control manifest sha256 is not locked for release: $BUNDLE_SHA" >&2
    exit 1
    ;;
esac
if [ "${#BUNDLE_SHA}" -ne 64 ]; then
  echo "Control manifest sha256 must be 64 hex chars" >&2
  exit 1
fi
case "$BUNDLE_SIZE" in
  ''|*[!0-9]*)
    echo "Control manifest size is invalid: $BUNDLE_SIZE" >&2
    exit 1
    ;;
esac
if [ "$BUNDLE_SIZE" -le 0 ]; then
  echo "Control manifest size must be positive" >&2
  exit 1
fi

ROOT="$CACHE/$BUNDLE_SHA"
ARCHIVE="$CACHE/$BUNDLE_SHA.tar.gz"
mkdir -p "$CACHE"

verify_file() {
  target="$1"
  [ -f "$target" ] &&
    [ "$(wc -c < "$target" | tr -d ' ')" = "$BUNDLE_SIZE" ] &&
    [ "$(sha256sum "$target" | cut -d ' ' -f 1)" = "$BUNDLE_SHA" ]
}

verify_root() {
  [ -f "$ROOT/.complete" ] &&
    [ -x "$ROOT/$BIN_NAME" ] &&
    { [ -z "$BIN_SHA" ] || [ "$(sha256sum "$ROOT/$BIN_NAME" | cut -d ' ' -f 1)" = "$BIN_SHA" ]; }
}

if ! verify_file "$ARCHIVE"; then
  if [ -n "${QWEN35_9B_DEV_CONTROL_ARCHIVE:-}" ] && verify_file "$QWEN35_9B_DEV_CONTROL_ARCHIVE"; then
    cp "$QWEN35_9B_DEV_CONTROL_ARCHIVE" "$ARCHIVE"
  else
    temp="$CACHE/.download-$$"
    trap 'rm -f "$temp" "$temp".*; rm -rf "$CACHE/.unpack-$$"' EXIT HUP INT TERM
    success=false
    for url in "$MS_URL" "$HF_URL"; do
      case "$url" in
        https://*) ;;
        *) continue ;;
      esac
      echo "Downloading verified Qwen 3.5 9B Defiant Fable controller"
      attempt=0
      while [ "$attempt" -lt 4 ]; do
        if wget -q -c -T 90 -O "$temp" "$url" && verify_file "$temp"; then
          mv "$temp" "$ARCHIVE"
          success=true
          break
        fi
        if [ -f "$temp" ] && [ "$(wc -c < "$temp" | tr -d ' ')" -ge "$BUNDLE_SIZE" ]; then
          rm -f "$temp"
        fi
        attempt=$(( attempt + 1 ))
        [ "$attempt" -ge 4 ] || sleep 2
      done
      [ "$success" = false ] || break
      rm -f "$temp"
    done
    if [ "$success" != true ]; then
      echo "Controller download or size/SHA256 validation failed" >&2
      exit 1
    fi
  fi
fi

if ! verify_root; then
  temp_root="$CACHE/.unpack-$$"
  trap 'rm -rf "$temp_root"' EXIT HUP INT TERM
  mkdir -p "$temp_root"
  tar -tzf "$ARCHIVE" > "$temp_root/list"
  if awk '/^\// || /(^|\/)\.\.($|\/)/ {bad=1} END {exit !bad}' "$temp_root/list"; then
    echo "Invalid controller archive path" >&2
    exit 1
  fi
  rm -f "$temp_root/list"
  tar -xzf "$ARCHIVE" -C "$temp_root"
  [ -x "$temp_root/$BIN_NAME" ]
  if [ -n "$BIN_SHA" ]; then
    [ "$(sha256sum "$temp_root/$BIN_NAME" | cut -d ' ' -f 1)" = "$BIN_SHA" ]
  fi
  [ "$(find "$temp_root" -type l | wc -l | tr -d ' ')" = 0 ]
  touch "$temp_root/.complete"
  rm -rf "$ROOT"
  mv "$temp_root" "$ROOT"
  verify_root || exit 1
fi

export QWEN35_9B_CONTROLLER_SHA="$BUNDLE_SHA"
exec "$ROOT/$BIN_NAME"
