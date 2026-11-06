"""python -m app.privacy <file.pdf|file.txt|->: print the de-identified text."""

import sys

from .deid import main

raise SystemExit(main(sys.argv[1:]))
