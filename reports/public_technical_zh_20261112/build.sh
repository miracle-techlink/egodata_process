#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for name in public_hand_coordinates public_surface_motion public_interaction_demo; do
  (cd figures && xelatex -interaction=nonstopmode -halt-on-error "$name.tex" > "build_$name.log")
done
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log
bibtex main > build_bibtex.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log
cp main.pdf egodata-technical-report-zh.pdf
cp main.pdf 完整技术报告_单栏公开版.pdf
