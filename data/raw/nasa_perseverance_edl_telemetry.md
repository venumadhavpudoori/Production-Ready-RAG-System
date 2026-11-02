---
document_id: NASA-M2020-EDL
title: Mars 2020 Perseverance Entry, Descent, and Landing (EDL) Flight Reconstruction and Aerothermal Performance
author: NASA Jet Propulsion Laboratory (JPL) & Langley Research Center (LaRC)
publication_year: 2021
mission: Mars 2020
target_body: Mars
landing_site: Jezero Crater (18.38°N, 77.58°E)
source_url: https://ntrs.nasa.gov/citations/20210018520
---

# Mars 2020 Perseverance Entry, Descent, and Landing (EDL) Flight Reconstruction and Aerothermal Performance

## 1. Executive Mission Overview
On February 18, 2021, NASA's Mars 2020 mission successfully delivered the Perseverance rover and the Ingenuity Mars Helicopter to the surface of Jezero Crater on Mars. The landing footprint achieved was approximately 7.7 km by 6.6 km, representing the most precise landing on the Martian surface to date. Atmospheric entry occurred at an inertial velocity of 5.37 km/s (approx. 19,300 km/h) at a flight path angle of -12.05 degrees at the Martian atmospheric interface (defined at a radial distance of 3,522.2 km from Mars center of mass).

## 2. Atmospheric Entry & Hypersonic Deceleration
During the hypersonic entry phase, the aeroshell protected the rover vehicle using a 70-degree sphere-cone heatshield made of Phenolic-Impregnated Carbon Ablator (PICA) with a diameter of 4.5 meters. Aerodynamic deceleration reached a peak deceleration of 12.5 Earth g's approximately 80 seconds after atmospheric entry. The entry capsule utilized guided entry via roll control thrusters (reaction control system, RCS) running on monopropellant hydrazine ($N_2H_4$) to fly a lifting trajectory with an aerodynamic lift-to-drag ($L/D$) ratio of 0.24. This active guidance compensated for atmospheric density variations measured in real time.

## 3. MEDLI2 Sensor Suite Instrumentation & Thermal Telemetry
The Mars Entry, Descent, and Landing Instrumentation 2 (MEDLI2) suite gathered extensive aerothermal, aerodynamic, and thermal protection system (TPS) flight performance data.
- **Sensor Compliment:** MEDLI2 consisted of 28 total sensors: 11 thermocouple plugs (MEP), 8 pressure transducers (MP), 3 heat flux sensors, and a thermal radiometer on the forebody heatshield, plus 6 sensors on the aftbody backshell (the first ever instrumented backshell on a Mars mission).
- **Peak Heatshield Heating & Pressure:** The maximum surface pressure recorded by the stagnation point transducer (MP01) was 36.4 kPa during the hypersonic pulse. The peak convective heat flux recorded was 118 W/cm², while radiative heating contributed an additional 14 W/cm², totaling a peak heat flux of 132 W/cm².
- **Peak Backshell Heating:** The instrumented backshell experienced a maximum radiative and convective heating pulse of 18.2 W/cm² in the wake separation zone, well within the SIRCA (Silicone-Impregnated Refractory Ceramic Coating) thermal limits.
- **Subsurface Temperatures:** Deep thermocouple data indicated a peak PICA bondline temperature of 142°C, maintaining the inner composite aeroshell well below the structural allowable limit of 250°C.

## 4. Supersonic Parachute Deployment & Range Trigger
A critical advancement in the Mars 2020 EDL sequence was the Range Trigger algorithm.
- **Range Trigger Activation:** Unlike MSL Curiosity, which deployed its parachute at a pre-calculated velocity, Perseverance continuously calculated its remaining downrange distance to the targeted landing ellipse. Range Trigger triggered mortar firing at a downrange distance of 9.8 km from the target center at Mission Elapsed Time (MET) E+242.4 seconds.
- **Parachute Flight Conditions:** The 21.5-meter diameter Disk-Gap-Band (DGB) supersonic parachute deployed at a velocity of Mach 1.83 (dynamic pressure $q = 412$ Pa) at an altitude of 11.2 km above the Martian terrain. The peak deployment inflation load measured was 147 kN, comfortably below the 300 kN structural load capability validated during the ASPIRE sounding rocket tests.

## 5. Terrain-Relative Navigation (TRN) & Divert Maneuver
Perseverance utilized the Lander Vision System (LVS) implementing Terrain-Relative Navigation (TRN).
- **LVS Operation:** Operating between altitudes of 4.2 km and 1.5 km during parachute descent, the downward-looking optical descent camera took 1024x1024 pixel images at 10 Hz. The onboard FPGA matched real-time surface features against an orbital map of Jezero Crater pre-loaded in memory.
- **Position Accuracy:** TRN determined the rover's position with an absolute horizontal uncertainty of less than 40 meters within 10 seconds of camera activation.
- **Divert Decision:** The Safe Target Selection (STS) algorithm identified hazardous boulder fields and crater scarps near the original landing zone and computed a divert maneuver of 312 meters to a safe, level patch of Jezero delta sediments.

## 6. Powered Descent & Sky Crane Terminal Landing
- **Descent Stage Ignition:** At an altitude of 2.1 km above ground level and a velocity of 88 m/s, the rover and descent stage separated from the backshell and parachute. Eight throttleable Mars Lander Engines (MLE), each producing up to 3.1 kN of thrust using monopropellant hydrazine, decelerated the vehicle to a constant vertical velocity of 0.75 m/s.
- **Sky Crane Deployment:** At 21.4 meters above the ground, the Sky Crane maneuver commenced. Three Kevlar bridles and an electrical umbilical lowered the 1,025 kg rover 7.6 meters below the descent stage.
- **Touchdown Telemetry:** Perseverance wheels contacted the Martian surface at 20:55 UTC on Feb 18, 2021. The bridles were severed via pyrotechnic cable cutters at touchdown detection, and the descent stage performed an autonomous flyaway throttle burn, impacting the surface 700 meters to the northwest.

