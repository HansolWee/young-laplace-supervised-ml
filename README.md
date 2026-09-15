# Young–Laplace Supervised ML

A supervised neural surrogate for predicting axisymmetric liquid-bridge equilibrium solution governed by the Young–Laplace equation.

A Fortran finite-element Newton solver generates reference solutions. A PyTorch model learns the bridge geometry and reference pressure across Bond numbers, then evaluates predictions at held-out Bond numbers.

## Overview

The workflow consists of three stages:

1. Generate numerical reference solutions with the Fortran solver.
2. Train the neural surrogate on selected Bond numbers.
3. Compare predictions with reference solutions at unseen Bond numbers.

The training objective uses solution data only. No governing-equation residual is included in the neural-network loss, and no Fortran calls are made during optimization.

## Physical problem

The bridge has fixed contact lines, unit end radii, and dimensionless length $L=2$:

$$
r(0)=r(L)=1.
$$

Its prescribed volume is

$$
V=\pi\int_0^L r(z)^2\,dz=2\pi.
$$

With dimensionless surface tension equal to one, the Young–Laplace equation for a profile representable as $r(z)$ is

$$
\frac{1}{r\sqrt{1+r_z^2}}
-
\frac{r_{zz}}{(1+r_z^2)^{3/2}}
=
P_0+\mathrm{Bo}\,z.
$$

The Fortran implementation uses a parametric representation of the interface and solves for its coordinates and the unknown pressure $P_0$.

## Neural surrogate

The model contains two tanh networks:

- **Geometry network:** computational coordinate $t\in[0,1]$ and Bond number → radial and axial corrections.
- **Pressure network:** Bond number → reference pressure.

The outputs are constructed as

$$
r(t,\mathrm{Bo})=\exp(N_r(t,\mathrm{Bo})),
$$

$$
z(t,\mathrm{Bo})=2t+N_z(t,\mathrm{Bo}),
$$

$$
P_0(\mathrm{Bo})=1-\mathrm{Bo}+N_p(\mathrm{Bo}).
$$

The exponential representation enforces positive radius. Boundary conditions and volume conservation are learned from the reference data rather than imposed exactly by this network architecture.

### Training objective

For the solution vector

$$
\mathbf{U}=(r_1,z_1,\ldots,r_N,z_N,P_0),
$$

the supervised loss is

$$
\mathcal{L}
=
\frac{1}{N_B}
\sum_{b=1}^{N_B}
\frac{1}{2}
\left\|
\mathbf{U}_{\theta,b}
-
\mathbf{U}_{\mathrm{Newton},b}
\right\|_2^2.
$$

Training uses float64 arithmetic, Adam optimization, and scaled L-BFGS refinement. The best recorded model is retained.

The loss is averaged over training Bond numbers and summed over solution components.

## Installation

Requirements:

- Python with PyTorch, NumPy, and Matplotlib
- GNU Fortran (`gfortran`)
- GNU Make

Clone the repository:

```bash
git clone https://github.com/HansolWee/young-laplace-supervised-ml.git
cd young-laplace-supervised-ml
```

Install Python dependencies:

```bash
python -m pip install torch numpy matplotlib
```

Build the Fortran executables:

```bash
make
```

This builds:

- `yl`: Newton reference solver
- `yl_evaluate`: residual and gradient evaluator

The supervised training script uses `yl`.

## Usage

### Train with default settings

```bash
python train_yl_supervised.py --output yl_supervised
```

By default, the script uses:

- 10 training Bond numbers evenly spaced from 0.1 to 2.8
- 9 test Bond numbers evenly spaced from 0.25 to 2.65
- Up to 5,000 Adam updates
- Up to 200 outer L-BFGS steps
- An unscaled data-loss target of `1e-10`
- Random seed 7

Each outer L-BFGS step may perform multiple internal iterations and loss evaluations. Optimization may stop early when the loss target is reached.

Use a new or empty output directory for each run.

### Choose training and test cases

```bash
python train_yl_supervised.py \
    --train-bonds 0.1 0.4 0.7 1.0 1.3 1.6 1.9 2.2 2.5 2.8 \
    --test-bonds 0.25 0.55 0.85 1.15 1.45 1.75 2.05 2.35 2.65 \
    --output results_custom
```

Training and test Bond numbers must be distinct and lie within the configured normalization interval.

List all options:

```bash
python train_yl_supervised.py --help
```

### Run the reference solver alone

```bash
./yl 100 0.1
```

The arguments specify the number of finite elements and the Bond number. The solver writes `u_newton.dat` in the current directory.

## Outputs

The training script saves:

| File | Contents |
|---|---|
| `config.json` | Run settings and model metadata |
| `model.pt` | Selected model weights and metadata |
| `history.csv` | Recorded optimization history |
| `history.png` | Supervised loss plot |
| `train_metrics.csv` | Errors on training cases |
| `test_metrics.csv` | Errors on held-out cases |
| `test_max_errors.json` | Maximum errors across test cases |
| `test_error_vs_bond.png` | Relative solution-vector error versus Bond number |

Per-case directories under `train/` and `test/` contain reference solutions, predictions, comparison figures, metrics, and Tecplot output.

Figures are saved automatically without requiring an interactive display.

## Validation and scope

Evaluation reports relative errors in the solution vector and coordinate fields, together with absolute and relative pressure errors.

The default test cases measure interpolation within the training Bond-number range. They do not establish extrapolation performance.

The reference solutions are numerical finite-element solutions. Agreement with them measures surrogate accuracy relative to that discretization.

The reported training tolerance applies to supervised data loss, not to the Young–Laplace residual.

## Coordinate and pressure conventions

This solver uses the pressure term

$$
P_0+\mathrm{Bo}\,z.
$$

The companion small-Bond-number perturbation PINN uses a pressure term of the form $K-Gz$.

For the same domain length and $G=\mathrm{Bo}$, these conventions are related by

$$
z_{\mathrm{PINN}}=L-z_{\mathrm{Fortran}},
\qquad
K=P_0+GL.
$$

Comparisons therefore require matching domain lengths, reversing the axial coordinate, and shifting the reference pressure.

The perturbation solution retains terms only through first order in the Bond number, whereas the Fortran solver solves the nonlinear problem.

## Source files

| File | Purpose |
|---|---|
| `train_yl_supervised.py` | Data generation, training, and evaluation |
| `mod_yl.f90` | Finite-element assembly and Newton solver |
| `mod_yl_benchmark.f90` | Problem setup and initial geometry |
| `main_yl.f90` | Reference-solver command-line program |
| `mod_yl_python.f90` | Residual and gradient evaluation |
| `main_yl_evaluate.f90` | Evaluator command-line program |
| `Makefile` | Fortran build rules |
