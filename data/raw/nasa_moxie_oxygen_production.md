---
document_id: NASA-M2020-MOXIE
title: Mars Oxygen In-Situ Resource Utilization Experiment (MOXIE) Flight Demonstration and Technical Operations
author: Massachusetts Institute of Technology (MIT) Haystack Observatory & NASA JPL
publication_year: 2022
mission: Mars 2020
target_body: Mars
payload_location: Rover Chassis Interior
source_url: https://ntrs.nasa.gov/citations/20220015944
---

# Mars Oxygen In-Situ Resource Utilization Experiment (MOXIE) Flight Demonstration and Technical Operations

## 1. System Objective and In-Situ Resource Utilization (ISRU) Overview
The Mars Oxygen ISRU Experiment (MOXIE) is a technology demonstration payload on board the Mars 2020 Perseverance rover designed to demonstrate the production of breathable oxygen ($O_2$) directly from the ambient Martian atmosphere, which is composed of roughly 95.3% carbon dioxide ($CO_2$), 2.6% molecular nitrogen ($N_2$), and 1.9% argon ($Ar$). Human Mars missions will require approximately 25 to 30 metric tons of oxygen to serve as propellant oxidizer for the Mars Ascent Vehicle (MAV) and life support for a 4-person crew. MOXIE serves as a 1:200 scale prototype of a full-scale automated production facility.

## 2. Atmospheric Ingestion and Compressor Subsystem
Ambient Martian atmospheric pressure varies seasonally between 600 Pa and 1000 Pa (0.006 to 0.010 bar) at surface level.
- **Scroll Compressor:** MOXIE pulls in outside Martian air through an ultra-low-pressure HEPA filter to eliminate airborne dust particles down to 0.3 micrometers. A high-speed mechanical scroll compressor rotating up to 3,500 RPM compresses the gas by a factor of roughly 100x, raising the ambient gas pressure to approximately 101.3 kPa (1 atmosphere absolute pressure).
- **Thermal Pre-Heating:** Before entering the chemical reactor, the pressurized $CO_2$ stream passes through a regenerative counterflow heat exchanger that reclaims heat from exiting exhaust streams, pre-heating the incoming gas to approximately 400°C.

## 3. Solid Oxide Electrolysis Cells (SOEC) Chemical Reaction Architecture
The heart of MOXIE is a stack of 10 Solid Oxide Electrolysis Cells (SOEC) manufactured by OxEon Energy using scandia-stabilized zirconia (ScSZ) ceramic electrolyte plates with mixed-conducting porous cermet electrodes.
- **Electrochemical Reaction:** Inside the SOEC at an operating temperature of 800°C (1,073 K), carbon dioxide molecules are catalytically dissociated at the porous porous cathode:
  $$2CO_2 + 4e^- \rightarrow 2CO + 2O^{2-}$$
- **Oxygen Ion Migration:** The $O^{2-}$ oxygen ions are transported across the solid ceramic electrolyte lattice via oxygen vacancies driven by an applied electric potential.
- **Anodic Recombination:** At the anode, oxygen ions are oxidized and recombined into pure molecular oxygen gas:
  $$2O^{2-} \rightarrow O_2 + 4e^-$$
- **Net Reaction:** The net chemical process is $2CO_2 \rightarrow 2CO + O_2$.

## 4. Production Rates, Purity, and Power Metrics
- **Oxygen Production Rate:** Across multiple test runs conducted over different seasons and Martian diurnal cycles (including nighttime and midday runs), MOXIE produced oxygen at steady-state rates between 6.0 and 12.2 grams per hour ($g/hr$).
- **Purity Level:** Luminescence and zirconia sensor analysis confirmed that the produced oxygen reached a measured chemical purity of $\ge 98.0\%$, with typical purity exceeding 99.5%, containing trace harmless argon and nitrogen impurities and zero toxic carbon monoxide bypass.
- **Power Consumption:** MOXIE draws approximately 300 Watts of electrical power during steady-state operation from the rover's Multi-Mission Radioisotope Thermoelectric Generator (MMRTG) and lithium-ion battery buffer. 
- **Thermal Management:** Because the SOEC operates at 800°C while the surrounding rover interior must remain within -40°C to +50°C, the SOEC is encased in an aerogel thermal insulation vacuum enclosure surrounded by a gold-coated exterior jacket that reflects 98% of radiant thermal energy.

## 5. Coking Prevention and Operational Constraints
A primary operational limit in SOEC electrolysis is avoiding "coking" (the irreversible deposition of solid carbon soot onto the cathode catalyst):
- **Boudouard Reaction Threshold:** If the applied electrical voltage is driven too high, or the local $CO_2$ flow rate drops, carbon monoxide reduces further to solid graphite:
  $$2CO \rightarrow C_{(s)} + CO_2$$
- **Voltage Clamping Guardrail:** MOXIE’s embedded control firmware continuously monitors the Nernst potential across each individual cell. The system strictly clamps the operating voltage to ensure cell potential never drops below the coking boundary of approximately -0.92 V against reference, maintaining cathode integrity throughout its planned 16 operational test campaigns.

