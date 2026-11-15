#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for name in hand_pose_principle pose6d_principle stereo_geometry_principle; do
  (cd figures && xelatex -interaction=nonstopmode -halt-on-error "$name.tex" > "build_$name.log")
done
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log
bibtex main > build_bibtex.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log
xelatex -interaction=nonstopmode -halt-on-error demo_materials.tex > build_demo1.log
xelatex -interaction=nonstopmode -halt-on-error demo_materials.tex > build_demo2.log
xelatex -interaction=nonstopmode -halt-on-error visual_check.tex > build_visual.log
cp main.pdf 完整技术报告_手部与6D配图版.pdf
cp visual_check.pdf 手部与6D_三页配图预览.pdf
