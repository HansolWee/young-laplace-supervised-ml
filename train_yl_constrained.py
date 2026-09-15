"""Physics-constrained data-driven neural surrogate.
Newton labels are generated before optimization; no Fortran calls in training.
Loss = data_loss + lambda_bc * bc_loss + lambda_volume * volume_loss.
Constraints use training Bo only; no hard constraint or inference projection.
"""

from pathlib import Path
import argparse
import csv
import json
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Neural surrogate model
# ============================================================

class BridgeNN(nn.Module):
    def __init__(
        self,
        width,
        bond_min,
        bond_max,
        depth=2,
        pressure_depth=1,
    ):
        super().__init__()

        # Bond numbers are normalized to [-1, 1], so the interval must be valid.
        if bond_max <= bond_min:
            raise ValueError("bond_max must exceed bond_min")

        # Store normalization limits as buffers so they follow the model state.
        self.register_buffer(
            "bond_min",
            torch.tensor(float(bond_min)),
        )
        self.register_buffer(
            "bond_max",
            torch.tensor(float(bond_max)),
        )

        # All network dimensions must be strictly positive.
        if width < 1 or depth < 1 or pressure_depth < 1:
            raise ValueError(
                "width, depth and pressure_depth must be positive"
            )

        def build_mlp(input_size, output_size, hidden_depth):
            layers = []
            previous_size = input_size

            # Hidden layers: Linear -> Tanh.
            for _ in range(hidden_depth):
                layers.append(nn.Linear(previous_size, width))
                layers.append(nn.Tanh())
                previous_size = width

            # Final linear layer produces the unconstrained network output.
            layers.append(nn.Linear(previous_size, output_size))
            return nn.Sequential(*layers)

        # Field network:
        #     (t, Bo) -> corrections to the reference r-z shape.
        self.net = build_mlp(
            input_size=2,
            output_size=2,
            hidden_depth=depth,
        )

        # Pressure network:
        #     Bo -> pressure correction.
        self.pressure_net = build_mlp(
            input_size=1,
            output_size=1,
            hidden_depth=pressure_depth,
        )

        # Start close to the reference state:
        #     r = 1, z = 2t, P0 = 1 - Bo.
        for network in (self.net, self.pressure_net):
            nn.init.normal_(network[-1].weight, std=1.0e-3)
            nn.init.zeros_(network[-1].bias)

    def forward(self, t, bond):
        # Ensure the coordinate is stored as a column vector.
        t = t.reshape(-1, 1)

        # Normalize the Bond number from [bond_min, bond_max] to [-1, 1].
        b = t.new_full((1, 1), float(bond))
        b_scaled = (
            2.0 * (b - self.bond_min)
            / (self.bond_max - self.bond_min)
            - 1.0
        )

        # Use normalized position and normalized Bond number as field inputs.
        inputs = torch.cat(
            (2.0 * t - 1.0, b_scaled.expand_as(t)),
            dim=1,
        )

        # Predict corrections to the reference bridge shape.
        correction = self.net(inputs)

        # Exponential parameterization keeps the radius positive.
        r = torch.exp(correction[:, 0])
        z = 2.0 * t[:, 0] + correction[:, 1]

        # One pressure value is predicted for each Bond number and shared
        # across all spatial nodes.
        p0 = (
            1.0 - float(bond)
            + self.pressure_net(b_scaled).reshape(1)
        )

        # Flatten the nodal (r, z) pairs and append the scalar pressure.
        rz = torch.stack((r, z), dim=1).reshape(-1)
        return torch.cat((rz, p0))


# ============================================================
# Utility functions
# ============================================================

def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def generate_label(executable, nelem, bond, folder, timeout):
    folder.mkdir(parents=True)

    # Run each Newton solve in its own temporary working directory.
    with tempfile.TemporaryDirectory(
        prefix="yl_supervised_newton_"
    ) as directory:
        try:
            result = subprocess.run(
                [str(executable), str(nelem), f"{bond:.17e}"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:

            def decoded(value):
                return (
                    value.decode(errors="replace")
                    if isinstance(value, bytes)
                    else (value or "")
                )

            # Preserve solver output to make timeout failures diagnosable.
            (folder / "newton.log").write_text(
                decoded(exc.stdout) + decoded(exc.stderr)
            )
            raise RuntimeError(
                f"Newton timeout at Bo={bond}; see {folder}"
            ) from exc

        # Save stdout and stderr from the Newton solve.
        (folder / "newton.log").write_text(
            result.stdout + "\n" + result.stderr
        )

        if result.returncode:
            raise RuntimeError(
                f"Newton failed at Bo={bond}; "
                f"see {folder / 'newton.log'}"
            )

        u = np.loadtxt(
            Path(directory) / "u_newton.dat",
            ndmin=1,
        )

    # Verify the expected solution-vector size and numerical validity.
    if u.shape != (4 * nelem + 3,) or not np.isfinite(u).all():
        raise ValueError(f"Invalid Newton label at Bo={bond}")

    np.savetxt(folder / "u_newton.dat", u, fmt="%.17e")
    return torch.tensor(u, dtype=torch.float64)


# ============================================================
# Loss terms and physical constraints
# ============================================================

def data_loss(model, t, bonds, targets):
    predictions = torch.stack([model(t, b) for b in bonds])
    return 0.5 * (predictions - targets).square().sum(dim=1).mean()


def bc_residual(u):
    """Essential BCs in Fortran order: r0, z0, rf, zf."""

    return torch.stack(
        (
            u[..., 0] - 1,
            u[..., 1],
            u[..., -3] - 1,
            u[..., -2] - 2,
        ),
        dim=-1,
    )


def fem_volume(u):
    """Same quadratic elements and 3-point Gauss rule as mod_yl.f90.

    Local coordinate is [0,1]; z derivative already includes element geometry.
    Do not multiply by a separate element length or take abs(dz/dxi).
    """

    # Extract nodal radius and axial-coordinate values from the interleaved
    # solution vector. The last entry is pressure and is excluded here.
    r, z = u[..., :-1:2], u[..., 1:-1:2]

    if r.shape[-1] < 3 or r.shape[-1] % 2 != 1:
        raise ValueError("Expected quadratic FEM nodes")

    # Three-point Gauss quadrature mapped to the local interval [0, 1].
    x = u.new_tensor(
        [
            (1 - .774596669241483) * .5,
            .5,
            (1 + .774596669241483) * .5,
        ]
    )
    w = u.new_tensor(
        [
            .555555555555556 * .5,
            .888888888888889 * .5,
            .555555555555556 * .5,
        ]
    )

    # Quadratic shape functions and their derivatives.
    phi = torch.stack(
        (
            1 - 3 * x + 2 * x * x,
            4 * (x - x * x),
            -x + 2 * x * x,
        )
    )
    dphi = torch.stack(
        (
            -3 + 4 * x,
            4 - 8 * x,
            -1 + 4 * x,
        )
    )

    # Evaluate radius and dz/dxi at the quadrature points element by element.
    rq = r.unfold(-1, 3, 2) @ phi
    zq = z.unfold(-1, 3, 2) @ dphi

    return torch.pi * (rq.square() * zq * w).sum(dim=(-2, -1))


def loss_terms(model, t, bonds, targets, args):
    predictions = torch.stack([model(t, b) for b in bonds])

    data = .5 * (predictions - targets).square().sum(dim=1).mean()
    bc = .5 * bc_residual(predictions).square().sum(dim=1).mean()
    volume = .5 * (fem_volume(predictions) - 2 * torch.pi).square().mean()

    total = (
        data
        + args.lambda_bc * bc
        + args.lambda_volume * volume
    )

    return total, data, bc, volume


def constraint_metrics(u):
    value = torch.as_tensor(u, dtype=torch.float64)
    b = bc_residual(value)
    volume = float(fem_volume(value))

    return dict(
        BC_norm=float(torch.linalg.vector_norm(b)),
        BC_max_abs=float(b.abs().max()),
        volume=volume,
        volume_error=volume - 2 * np.pi,
        volume_absolute_error=abs(volume - 2 * np.pi),
        volume_relative_error=abs(volume - 2 * np.pi) / (2 * np.pi),
    )


# ============================================================
# Optimization
# ============================================================

def optimize(model, t, bonds, targets, args):
    history = []
    best = float("inf")
    best_state = snapshot(model)

    def record(stage, step):
        nonlocal best, best_state

        with torch.no_grad():
            terms = [
                v.item()
                for v in loss_terms(model, t, bonds, targets, args)
            ]
            value = terms[0]

        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite supervised loss")

        history.append((len(history), stage, step, *terms))

        if value < best:
            best, best_state = value, snapshot(model)

        if step % args.log_every == 0 or stage == "LBFGS":
            print(
                f"{stage} step={step} total_loss={value:.12e} "
                f"data_loss={terms[1]:.12e} "
                f"bc_loss={terms[2]:.12e} "
                f"volume_loss={terms[3]:.12e}",
                flush=True,
            )

        return value

    def backward(loss):
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite objective")

        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all().item():
                raise FloatingPointError(f"Invalid gradient: {name}")

    # --------------------------------------------------------
    # Stage 1: Adam
    # --------------------------------------------------------

    model.train()
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    value = record("Adam", 0)

    for step in range(1, args.steps + 1):
        if value <= args.tol:
            break

        adam.zero_grad(set_to_none=True)
        backward(loss_terms(model, t, bonds, targets, args)[0])
        adam.step()
        value = record("Adam", step)

    # Restore the best state encountered during Adam.
    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Stage 2: L-BFGS
    # --------------------------------------------------------

    if args.lbfgs_steps and best > args.tol:
        # Scale the objective using the best loss from the Adam stage.
        scale = 1.0 / max(best, 1.0e-12)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=20,
            max_eval=40,
            history_size=100,
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        calls = 0

        def closure():
            nonlocal calls

            lbfgs.zero_grad(set_to_none=True)
            loss = scale * loss_terms(model, t, bonds, targets, args)[0]
            backward(loss)
            calls += 1
            return loss

        print(
            f"Starting constrained L-BFGS, fixed scale={scale:.6e}",
            flush=True,
        )

        for step in range(1, args.lbfgs_steps + 1):
            lbfgs.step(closure)
            value = record("LBFGS", step)

            if value <= args.tol:
                break

        print(f"L-BFGS closure calls: {calls}")

    # Always return the best model state found across both stages.
    model.load_state_dict(best_state)
    return history, best


# ============================================================
# Error metrics and output comparison
# ============================================================

def relative_error(pred, ref):
    denominator = float(np.linalg.norm(ref))
    return (
        float(np.linalg.norm(pred - ref) / denominator)
        if denominator > 1e-14
        else None
    )


def compare(model, t, bond, target, folder):
    with torch.no_grad():
        u = model(t, bond).numpy()

    ref = target.numpy()

    if not np.isfinite(u).all():
        raise FloatingPointError(f"Nonfinite prediction at Bo={bond}")

    rz, exact = (
        u[:-1].reshape(-1, 2),
        ref[:-1].reshape(-1, 2),
    )

    metrics = dict(
        bond=float(bond),
        relative_U_error=relative_error(u, ref),
        relative_r_error=relative_error(rz[:, 0], exact[:, 0]),
        relative_z_error=relative_error(rz[:, 1], exact[:, 1]),
        P0_absolute_error=float(abs(u[-1] - ref[-1])),
        P0_relative_error=relative_error(u[-1:], ref[-1:]),
        NN_P0=float(u[-1]),
        Newton_P0=float(ref[-1]),
    )

    # Report constraint errors for both the neural and Newton solutions.
    metrics.update(
        {"NN_" + k: v for k, v in constraint_metrics(u).items()}
    )
    metrics.update(
        {"Newton_" + k: v for k, v in constraint_metrics(ref).items()}
    )

    # Save the neural solution and scalar metrics.
    np.savetxt(folder / "u_nn.dat", u, fmt="%.17e")
    (folder / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False)
    )
    (folder / "metrics.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in metrics.items()) + "\n"
    )

    # Write Newton and neural bridge shapes in Tecplot point format.
    with (folder / "comparison_tecplot.dat").open("w") as stream:
        stream.write(
            f'TITLE = "Constrained surrogate, Bo={bond:g}"\n'
            'VARIABLES = "z", "r"\n'
        )

        for name, xy in (("Newton", exact), ("NN constrained", rz)):
            stream.write(
                f'ZONE T="{name}", I={len(xy)}, DATAPACKING=POINT\n'
            )
            np.savetxt(stream, xy[:, [1, 0]], fmt="%.17e")

    # Plot the Newton reference and constrained neural prediction.
    fig, ax = plt.subplots()
    ax.plot(exact[:, 1], exact[:, 0], label="Newton")
    ax.plot(rz[:, 1], rz[:, 0], "--", label="Constrained NN")
    ax.set(
        xlabel="z",
        ylabel="r",
        title=f"Bo={bond:g}, L=2, V0=2π",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=200)
    plt.close(fig)

    print(
        f"Bo={bond:g}: "
        f"relative U error={metrics['relative_U_error']:.6e}"
    )

    return metrics


# ============================================================
# Main program
# ============================================================

def main():
    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    for name, default in (
        ("nelem", 100),
        ("width", 32),
        ("depth", 2),
        ("pressure-depth", 1),
        ("steps", 5000),
        ("lbfgs-steps", 200),
        ("log-every", 500),
        ("seed", 7),
    ):
        parser.add_argument("--" + name, type=int, default=default)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Unscaled total loss target; NOT a residual tolerance",
    )
    parser.add_argument("--bond-min", type=float, default=0.1)
    parser.add_argument("--bond-max", type=float, default=2.8)
    parser.add_argument(
        "--train-bonds",
        type=float,
        nargs="+",
        default=np.linspace(.1, 2.8, 10).tolist(),
    )
    parser.add_argument(
        "--test-bonds",
        type=float,
        nargs="+",
        default=np.linspace(.25, 2.65, 9).tolist(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yl_constrained"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Load compatible weights; start fresh optimizers",
    )
    parser.add_argument(
        "--newton",
        type=Path,
        default=Path(__file__).resolve().with_name("yl"),
    )
    parser.add_argument(
        "--newton-timeout",
        type=float,
        default=120,
    )
    parser.add_argument(
        "--lambda-bc",
        type=float,
        default=1.0,
        help="Weight of mean 0.5*sum(endpoint errors squared); default 1",
    )
    parser.add_argument(
        "--lambda-volume",
        type=float,
        default=1.0,
        help="Weight of mean 0.5*(FEM volume-2*pi)^2; default 1",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate command-line inputs
    # --------------------------------------------------------

    if (
        not np.isfinite([args.lambda_bc, args.lambda_volume]).all()
        or min(args.lambda_bc, args.lambda_volume) < 0
    ):
        parser.error("Penalty weights must be finite and nonnegative")

    if (
        min(
            args.nelem,
            args.width,
            args.depth,
            args.pressure_depth,
            args.log_every,
        ) < 1
        or min(args.steps, args.lbfgs_steps) < 0
    ):
        parser.error("Invalid integer settings")

    if (
        not np.isfinite(
            [
                args.lr,
                args.tol,
                args.bond_min,
                args.bond_max,
                args.newton_timeout,
            ]
        ).all()
        or min(args.lr, args.tol, args.newton_timeout) <= 0
        or args.bond_min >= args.bond_max
    ):
        parser.error("Invalid numeric settings")

    all_bonds = args.train_bonds + args.test_bonds

    if (
        not np.isfinite(all_bonds).all()
        or min(all_bonds) < args.bond_min
        or max(all_bonds) > args.bond_max
    ):
        parser.error(
            "Bond numbers must be finite and within the normalization interval"
        )

    if any(
        np.isclose(a, b, rtol=0, atol=1e-12)
        for i, a in enumerate(all_bonds)
        for b in all_bonds[i + 1:]
    ):
        parser.error(
            "Bond numbers must be unique; train and test must not overlap"
        )

    executable = args.newton.resolve()
    if not executable.is_file():
        parser.error(f"Missing Newton executable: {executable}")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output folder")

    # --------------------------------------------------------
    # Reproducibility and model construction
    # --------------------------------------------------------

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    model = BridgeNN(
        args.width,
        args.bond_min,
        args.bond_max,
        args.depth,
        args.pressure_depth,
    ).double()

    metadata = dict(
        approach="Physics-constrained data-driven neural surrogate",
        model_type="ConditionalBridgeNN",
        nelem=args.nelem,
        width=args.width,
        depth=args.depth,
        pressure_depth=args.pressure_depth,
        length=2.0,
        volume=float(2 * np.pi),
        bond_min=args.bond_min,
        bond_max=args.bond_max,
        train_bonds=args.train_bonds,
        lambda_bc=args.lambda_bc,
        lambda_volume=args.lambda_volume,
        constraint_bonds=args.train_bonds,
    )

    # Optionally load a compatible model state before optimization.
    if args.resume:
        checkpoint = torch.load(
            args.resume.resolve(),
            map_location="cpu",
            weights_only=True,
        )

        for key, value in metadata.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint mismatch: {key}")

        model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------------
    # Generate Newton labels
    # --------------------------------------------------------

    output.mkdir(parents=True, exist_ok=True)

    (output / "config.json").write_text(
        json.dumps(
            {
                **metadata,
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            },
            indent=2,
        )
    )

    print(metadata["approach"], flush=True)

    datasets = {}

    for split, bonds in (
        ("train", args.train_bonds),
        ("test", args.test_bonds),
    ):
        dataset = []

        for i, bond in enumerate(bonds):
            folder = output / split / f"{i:03d}_Bo_{bond:.6f}"

            print(
                f"Generating {split} Newton label: Bo={bond:g}",
                flush=True,
            )

            dataset.append(
                (
                    bond,
                    generate_label(
                        executable,
                        args.nelem,
                        bond,
                        folder,
                        args.newton_timeout,
                    ),
                    folder,
                )
            )

        datasets[split] = dataset

    # --------------------------------------------------------
    # Train the neural surrogate
    # --------------------------------------------------------

    t = torch.linspace(
        0,
        1,
        2 * args.nelem + 1,
    ).reshape(-1, 1)

    targets = torch.stack(
        [item[1] for item in datasets["train"]]
    )

    print(
        "Dataset ready. Starting training; "
        "no Fortran calls during optimization.",
        flush=True,
    )

    history, best = optimize(
        model,
        t,
        args.train_bonds,
        targets,
        args,
    )

    model.eval()

    # --------------------------------------------------------
    # Save trained model and optimization history
    # --------------------------------------------------------

    torch.save(
        {
            **metadata,
            "model_state_dict": model.state_dict(),
            "best_total_loss": best,
            "optimizer_method": "Adam_then_scaled_LBFGS",
            "test_bonds": args.test_bonds,
        },
        output / "model.pt",
    )

    with (output / "history.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "record",
                "stage",
                "step",
                "total_loss",
                "data_loss",
                "bc_loss",
                "volume_loss",
            ]
        )
        writer.writerows(history)

    # Plot all recorded loss components.
    fig, ax = plt.subplots()

    for column, label in enumerate(
        (
            "total",
            "data",
            "BC (unweighted)",
            "volume (unweighted)",
        ),
        start=3,
    ):
        ax.semilogy(
            [r[0] for r in history],
            [
                max(r[column], np.finfo(float).tiny)
                for r in history
            ],
            label=label,
        )

    ax.legend()
    ax.set(
        xlabel="Recorded optimization step",
        ylabel="Loss",
    )
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "history.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # Evaluate train and test datasets
    # --------------------------------------------------------

    test_metrics = []

    for split, dataset in datasets.items():
        rows = [
            compare(model, t, bond, target, folder)
            for bond, target, folder in dataset
        ]

        with (output / f"{split}_metrics.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

        if split == "test":
            test_metrics = rows

    # Collect the worst test errors across all test Bond numbers.
    keys = [
        "relative_U_error",
        "relative_r_error",
        "relative_z_error",
        "P0_absolute_error",
        "P0_relative_error",
    ]
    keys += [
        "NN_BC_norm",
        "NN_BC_max_abs",
        "NN_volume_absolute_error",
        "NN_volume_relative_error",
    ]

    maxima = {
        k: max(
            (r[k] for r in test_metrics if r[k] is not None),
            default=None,
        )
        for k in keys
    }

    (output / "test_max_errors.json").write_text(
        json.dumps(maxima, indent=2, allow_nan=False)
    )

    # Plot the overall solution error versus Bond number.
    fig, ax = plt.subplots()
    ordered = sorted(test_metrics, key=lambda row: row["bond"])

    ax.semilogy(
        [r["bond"] for r in ordered],
        [
            max(r["relative_U_error"], np.finfo(float).tiny)
            for r in ordered
        ],
        "o-",
    )
    ax.set(
        xlabel="Bo",
        ylabel="Test relative U error",
    )
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "test_error_vs_bond.png", dpi=200)
    plt.close(fig)

    print(
        f"Selected total loss: {best:.12e}; "
        f"target met: {best <= args.tol}"
    )
    print("Maximum test errors:", maxima)
    print(f"Results written to: {output}")


if __name__ == "__main__":
    main()
