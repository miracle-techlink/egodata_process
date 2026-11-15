#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log
bibtex main > build_bibtex.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log
mkdir -p ../01_技术报告
cp main.pdf ../01_技术报告/手物交互预测_技术报告_精选修订版.pdf
