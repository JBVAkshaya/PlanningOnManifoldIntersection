# Constrained Nonlinear Kaczmarz Projection (cNKZ) on Intersection of Manifolds

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository hosts the open-source code for our paper:

> "Constrained Nonlinear Kaczmarz Projection on Intersections of Manifolds for Coordinated Multirobot Mobile Manipulation"

Constrained nonlinear Kaczmarz projection enables simultaneously satisfying various constraints, especially for a multi-robot team working in a tightly constrained fashion (e.g., moving a table).

## Installation

Clone the repository:
```bash
git clone https://github.com/JBVAkshaya/PlanningOnManifoldIntersection.git
```

## Repository Structure

### Core Components
- **`lib/`**: Core functionality
  - Projection algorithms
  - Constraint definitions
  - Collision avoidance algorithm
  - Plotting utilities
  - Other dependencies

- **`robot_configs/`**: Robot team and structure configurations
  - Different structures
  - Required team setups

- **`cellular_automata/`**: Environment generation
  - Library for generating environments

- **`algorithms/`**: Planning implementations
  - RRT with different projection techniques

- **`experiment_configs/`**: Experiment settings
  - Algorithmic parameters

## Usage

### Generate Environments
```bash
python scripts/generate_environments.py
```

### Run RRT with cNKZ
To run RRT with cNKZ on all generated environments:
```bash
python scripts/run_rrt_cnkz.py
```

## Contact

For questions or inquiries, please contact:
- [Akshaya Agrawal](https://github.com/JBVAkshaya)
