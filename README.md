# Damage-adaptive ChebyKAN deep energy method for anisotropic phase-field fracture

> **Status: submitted.** This repository is a partial release that contains the core components of the method. The complete code, including all example drivers, the finite element reference solver, input data and the scripts that reproduce the figures and tables, will be made publicly available here upon acceptance of the manuscript.

Manuscript:

> P. Zhang, P. Cui, K. Tang. *A Damage-Adaptive Mesh Deep Energy Method for Anisotropic Phase-Field Fracture*. Submitted to International Journal for Numerical Methods in Engineering.

The solver represents displacement and damage by a Chebyshev Kolmogorov–Arnold network (ChebyKAN) and minimizes the discrete phase-field energy on a variable-node quadrilateral quadtree. Refinement inherits the network weights, prolongates the irreversible history exactly between nested meshes, and re-solves each load step at the same load before it is accepted. For anisotropic materials the refined region is an ellipse aligned with the fiber direction.

## Contents of this partial release

| File | Purpose |
|---|---|
| `ChebyKAN_Q4_Adaptive.py` | Variable-node Q4 quadtree, Gupta transition basis, refinement with 2:1 balance, exact history prolongation |
| `ChebyKAN_Q4_Aniso.py` | Anisotropic phase-field model in material axes and fiber-aligned elliptic marker |
| `composite/dsolve.py` | Orthotropic material definitions |

## Requirements

Python 3.10 or newer, a CUDA GPU and the packages in `requirements.txt`.

## Usage

```bash
python ChebyKAN_Q4_Aniso.py --help
```

## Contact

Keke Tang (kktang@tongji.edu.cn), Pengfei Cui (pfcui@nuaa.edu.cn)
