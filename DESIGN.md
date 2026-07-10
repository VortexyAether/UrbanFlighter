# Urban Flighter Design System

## 1. Atmosphere & Identity

Urban Flighter is a dense urban-air-mobility command surface: technical, readable under pressure, and grounded in live simulator telemetry rather than marketing polish. The signature is glass-backed operational instrumentation over a full-bleed 2D/3D flight scene, with compact panels, tabular numbers, and clear source traceability.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/primary | --bg | #111316 | #111316 | Simulator background behind canvas |
| Surface/grid | --tile | rgba(255,255,255,0.055) | rgba(255,255,255,0.055) | Map/grid texture |
| Surface/panel | --panel | rgba(244,246,242,0.93) | rgba(244,246,242,0.93) | Main overlay panels |
| Surface/elevated | --panel-strong | rgba(255,255,255,0.97) | rgba(255,255,255,0.97) | Strong overlay states |
| Surface/dark | --dark-panel | rgba(20,22,24,0.82) | rgba(20,22,24,0.82) | Loading overlays |
| Text/primary | --ink | #111417 | #111417 | Panel headings and metrics |
| Text/secondary | --muted | #5d656c | #5d656c | Labels, captions, helper copy |
| Border/default | --line | rgba(17,20,23,0.16) | rgba(17,20,23,0.16) | Panel and control borders |
| Border/strong | --line-strong | rgba(17,20,23,0.34) | rgba(17,20,23,0.34) | Hover/active borders |
| Accent/primary | --blue | #22b8ff | #22b8ff | Live status, wind/cfd signal |
| Accent/primary-soft | --blue-soft | rgba(34,184,255,0.16) | rgba(34,184,255,0.16) | Live status surfaces |
| Accent/warning | --orange | #ff8a2a | #ff8a2a | Baseline/replay/training emphasis |
| Accent/warning-soft | --orange-soft | rgba(255,138,42,0.18) | rgba(255,138,42,0.18) | Baseline/replay surfaces |
| Status/error | --status-error | #8a2d12 | #8a2d12 | Inline error text |

### Rules

- Keep the UI neutral and data-first; blue means live model/wind, orange means baseline/replay/training readiness.
- Do not introduce decorative colors. Add a semantic token here before using a new color.
- Full-bleed simulator visuals carry the scene; panels stay compact and translucent.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Usage |
|-------|------|--------|-------------|----------|-------|
| H1 | 1.24rem | 800 | 1 | 0 | Panel title |
| H2 | 1.02rem | 800 | 1.05 | 0 | Metric values |
| H3 | 0.84rem | 800 | 1.2 | 0 | Small panel titles |
| Body | 0.78rem | 500 | 1.32 | 0 | Helper text, status |
| Caption | 0.68rem | 700 | 1.3 | 0.06em | Control sublabels |
| Overline | 0.66rem | 700 | 1.3 | 0.08em | Uppercase labels |

### Font Stack

- Primary: Space Grotesk, ui-sans-serif, system-ui, sans-serif
- Mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace

### Rules

- Simulator metrics use tabular numeric treatment where available.
- Do not scale typography with viewport width.
- Keep panel copy short enough to scan while flying or replaying.

## 4. Spacing & Layout

### Base Unit

All spacing derives from a base of 4px.

| Token | Value | Usage |
|-------|-------|-------|
| --space-1 | 4px | Tight inline gaps |
| --space-2 | 8px | Control gaps and compact cards |
| --space-3 | 12px | Panel offsets |
| --space-4 | 16px | Comfortable grouping |
| --space-5 | 20px | Larger panel separation |

### Grid

- Panels are fixed-width overlays: map panel around 318px, HUD around 338px.
- Wide desktop reserves the center for the 2D/3D simulator canvas.
- Mobile stacks map and HUD panels with constrained heights so the simulator remains visible.

### Rules

- Use 8px radius for panels, controls, and cards.
- Keep repeated metric cards in two-column grids unless the value needs the full row.
- Avoid nested cards; cards are for individual metrics, controls, and source readouts.

## 5. Components

### Overlay panel

- **Structure**: `aside.panel` with optional eyebrow, title, controls, and compact cards.
- **Variants**: left map panel, right HUD panel.
- **Spacing**: 8-10px internal padding, 8px gaps.
- **States**: scrollable if content exceeds viewport; no hidden controls.
- **Accessibility**: semantic `aside`, labels for grouped controls.
- **Motion**: only hover/active transform on controls.

### Metric card

- **Structure**: label plus strong tabular value.
- **Variants**: default, wide, accent.
- **Spacing**: 8px padding, 5px label/value gap.
- **States**: passive display only.
- **Accessibility**: text labels must name units or thresholds.
- **Motion**: none.

### Mode selector

- **Structure**: radio inputs inside labels with strong title and small source note.
- **Variants**: default blue model modes, orange real/baseline modes.
- **Spacing**: 7px padding, 8px gap.
- **States**: hover, active, disabled when a mode is not available.
- **Accessibility**: grouped by `aria-label`; radio inputs remain native.
- **Motion**: 150ms transform/border/background.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Micro | 150ms | ease | Button hover, mode hover |
| Standard | 200ms | ease-in-out | Panel/control state transitions |

### Rules

- Animate only transform, opacity, filter, background, and border color.
- Every button and selectable control needs hover and disabled states.
- Range sliders use native controls for predictable replay operation.

## 7. Depth & Surface

### Strategy

Mixed: translucent panel surfaces with borders and restrained shadows over the simulator canvas.

| Level | Value | Usage |
|-------|-------|-------|
| Panel shadow | 0 22px 58px rgba(0,0,0,0.38) | Main overlays |
| Command shadow | 0 20px 54px rgba(0,0,0,0.34) | Top command bar |
| Border | 1px solid var(--line) | Cards, controls, panels |

Depth must not distract from the flight surface. Prefer one clean panel layer and repeated metric cards over nested panels.
