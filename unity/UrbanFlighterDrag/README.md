# Unity drop-in for Urban Flighter drag

This is **not** a full Unity scene rewrite. The live product remains the React / Three.js cockpit.

Copy `Runtime/QuadraticAirDrag.cs` into a Unity project and call `QuadraticAirDrag.Integrate` every fixed physics step:

```
velocity = QuadraticAirDrag.Integrate(velocity, localWind, Time.fixedDeltaTime);
```

Same constants and formulas as:

- `frontend/src/simulation/quadraticAirDrag.ts`
- `backend/urban_flighter_physics/quadratic_air_drag.py`

Honest label: quadratic air-relative parasite drag + momentum-theory induced power. Not blade-element, not Navier-Stokes, not AirSim/Unity Physics default drag.
