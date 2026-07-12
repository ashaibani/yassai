#!/bin/bash
# Regenerate docs/presentation.pdf from docs/presentation.html as a full-bleed
# A4-landscape deck (no page margins, no header/footer). The @page rule in the
# HTML sets the 297mm x 210mm size and margin:0; these flags stop Chrome from
# overriding that with its print-dialog defaults.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML="$HERE/presentation.html"
PDF="$HERE/presentation.pdf"

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
[ -x "$CHROME" ] || CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=3000 \
  --print-to-pdf="$PDF" \
  "file://$HTML"

echo "wrote $PDF"
