# Urban Flighter preprint

Source: `urban_flighter.tex`

```bash
cd /Users/va/projects/Urban_Flighter
PYTHONPATH=backend python scripts/render_paper_figures.py
cd paper && tectonic urban_flighter.tex
```

PDF: `urban_flighter.pdf`

Figures in `figures/`:
- `fig_architecture.png` system split
- `fig_potential_flow.png` toy CFD-lite (not NS)
- `fig_power_curve.png` parasite vs induced
- `fig_wind_response.png` headwind / tailwind / stick-off
- `fig_trajectories.png` Gym fixture seed 10007
- `fig_metrics.png` Gym aggregate

Rebuild figures before changing numbers. This is a preprint draft, not an uploaded arXiv ID.
