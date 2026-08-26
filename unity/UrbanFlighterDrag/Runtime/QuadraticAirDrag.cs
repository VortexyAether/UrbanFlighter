using System;
using UnityEngine;

namespace UrbanFlighter.Drag
{
    /// <summary>
    /// Portable port of Urban Flighter quadratic air-relative drag.
    /// Same formulas as frontend/src/simulation/quadraticAirDrag.ts and
    /// backend/urban_flighter_physics/quadratic_air_drag.py.
    /// This is not blade-element theory and not Navier-Stokes.
    /// </summary>
    public static class QuadraticAirDrag
    {
        public const float AirDensityKgM3 = 1.225f;
        public const float DragCoefficient = 1.05f;
        public const float FrontalAreaM2 = 0.18f;
        public const float MassKg = 2.5f;
        public const float GravityMps2 = 9.81f;
        public const int RotorCount = 4;
        public const float PropellerDiameterM = 0.15f;
        public const float InducedPowerFactor = 1.15f;
        public const float AvionicsPowerW = 18f;
        public const float SensorPowerW = 8f;
        public const float LinearAirDragPerS = 0.28f;
        public const string ModelId = "quadratic-air-relative-v1";
        public const string Honesty =
            "QUADRATIC AIR-RELATIVE DRAG · MOMENTUM-THEORY INDUCED · NOT BLADE-ELEMENT / NOT NS";

        public static float ParasiteDragPerM()
        {
            return (0.5f * AirDensityKgM3 * DragCoefficient * FrontalAreaM2) / Mathf.Max(MassKg, 1e-9f);
        }

        public static float RotorDiskAreaM2()
        {
            float radius = PropellerDiameterM * 0.5f;
            return RotorCount * Mathf.PI * radius * radius;
        }

        public static float HoverInducedVelocityMps()
        {
            float weightN = MassKg * GravityMps2;
            return Mathf.Sqrt(weightN / (2f * AirDensityKgM3 * Mathf.Max(RotorDiskAreaM2(), 1e-9f)));
        }

        public static Vector3 Integrate(
            Vector3 groundVelocity,
            Vector3 windVelocity,
            float dt,
            float? kPerM = null,
            float? linearPerS = null)
        {
            if (!IsFinite(dt) || dt <= 0f) return groundVelocity;
            float k = kPerM ?? ParasiteDragPerM();
            float linear = linearPerS ?? LinearAirDragPerS;
            if (!(k > 0f)) k = 0f;
            if (!(linear > 0f)) linear = 0f;
            if (k == 0f && linear == 0f) return groundVelocity;
            Vector3 air = groundVelocity - windVelocity;
            float airSpeed = air.magnitude;
            float denom = 1f + dt * (k * airSpeed + linear);
            return windVelocity + air / denom;
        }

        public static DragPower EvaluatePower(Vector3 groundVelocity, Vector3 windVelocity)
        {
            Vector3 air = groundVelocity - windVelocity;
            float airSpeed = air.magnitude;
            float dragForceN = 0.5f * AirDensityKgM3 * DragCoefficient * FrontalAreaM2 * airSpeed * airSpeed;
            float parasitePowerW = dragForceN * airSpeed;
            float weightN = MassKg * GravityMps2;
            float inducedHover = HoverInducedVelocityMps();
            float inducedHoverW = InducedPowerFactor * weightN * inducedHover;
            float inducedPowerW = inducedHoverW / Mathf.Sqrt(1f + (airSpeed / Mathf.Max(inducedHover, 1e-6f)) * (airSpeed / Mathf.Max(inducedHover, 1e-6f)));
            float climbPowerW = Mathf.Max(0f, groundVelocity.y) * weightN;
            return new DragPower
            {
                RelativeAirSpeed = airSpeed,
                DragForceN = dragForceN,
                ParasitePowerW = parasitePowerW,
                InducedPowerW = inducedPowerW,
                ClimbPowerW = climbPowerW,
                TotalPowerW = AvionicsPowerW + SensorPowerW + parasitePowerW + inducedPowerW + climbPowerW,
            };
        }

        static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        [Serializable]
        public struct DragPower
        {
            public float RelativeAirSpeed;
            public float DragForceN;
            public float ParasitePowerW;
            public float InducedPowerW;
            public float ClimbPowerW;
            public float TotalPowerW;
        }
    }
}
