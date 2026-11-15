#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log
bibtex main > build_bibtex.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log
xelatex -interaction=nonstopmode -halt-on-error demo_materials.tex > build_demo1.log
xelatex -interaction=nonstopmode -halt-on-error demo_materials.tex > build_demo2.log
