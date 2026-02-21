# A Cycle-Level Distribution-Based Calibration of Microscopic Intersection Models Using UAV Trajectories
<p align="center">
  <img src="images/uranus.svg" alt="URANUS Project Logo" height="100">
</p>

<p align="center">
  <b>Funded by the ERC project URANUS:</b><br>
  <i>Real-Time Urban Mobility Management via Intelligent UAV-based Sensing</i>
</p>

---

## 📌 Overview

This repository contains the implementation and supplementary material for a **phase-relative, distribution-based calibration framework** for microscopic traffic simulation using UAV-derived vehicle trajectories.

The framework calibrates a SUMO intersection model by comparing **phase-relative exit time distributions** between observed UAV trajectories and simulated vehicle trajectories.

Instead of relying solely on aggregated traffic counts, the methodology evaluates the **temporal structure of vehicle departures within each signal phase**, enabling a higher-resolution and behaviorally consistent calibration.

---

## 🎯 Methodological Contribution

The proposed calibration framework:

- Segments UAV trajectory data into signal cycles  
- Constructs **phase-direction buckets**  
- Computes **phase-relative exit times**  
- Builds cumulative exit count curves  
- Evaluates discrepancies using a normalized percentage-based metric (nABC)  
- Optimizes simulation parameters using derivative-free optimization  

---

## 📂 Repository Structure
