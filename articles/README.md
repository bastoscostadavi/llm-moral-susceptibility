Draft paper for MLSys 2025
==========================

Files
- `articles/mlsys2025.tex`: LaTeX draft following the requested sectioning and an MLSys-style conditional.
- `articles/references.bib`: Placeholder bibliography (replace with real citations).
- `articles/mlsys2025.sty`, `articles/mlsys2025.bst`: Official MLSys 2025 style and bibliography style.

Compiling
- The draft auto-loads `mlsys2025.sty` if present; otherwise it falls back to a generic article layout so you can still iterate locally.
- Compile with: `pdflatex mlsys2025.tex && bibtex mlsys2025 && pdflatex mlsys2025 && pdflatex mlsys2025`.
- For submission, use the base style `\usepackage{mlsys2025}`. For camera-ready, switch to `\usepackage[accepted]{mlsys2025}`.

Notes
- The Results section contains placeholders for tables/figures; replace with actual outputs from your analyses in `results/` and `data/`.
- The Statistical Analysis section summarizes the planned metrics (susceptibility, robustness) and recommended tests; adapt as needed based on your final pipeline.

