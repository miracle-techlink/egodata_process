#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error 技术解读.tex
xelatex -interaction=nonstopmode -halt-on-error 技术解读.tex
