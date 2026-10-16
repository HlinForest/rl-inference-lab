"""D01: inspect storage, stride, views, and copies on CPU.

Run from the repository root:
    python -m kernels.tensor_layout --output reports/day01.json
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare live, nonempty, dense CPU tensors used in this lesson.

    This is an allocation comparison, not an overlapping-elements detector.
    It is intentionally not a general predicate for empty/meta/sparse tensors.
    """
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def build_examples() -> dict[str, torch.Tensor]:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transposed = base.t()
    stepped = base[:, ::2]
    return {
        "base": base,
        "transpose": transposed,
        "offset_slice": base[1:, 1:],
        "reshape_contiguous": base.reshape(-1),
        "reshape_transpose": transposed.reshape(-1),
        "step_slice": stepped,
        "reshape_step_slice": stepped.reshape(-1),
        "advanced_index": base[[0, 2]],
    }


def describe(tensor: torch.Tensor, base: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "shares_base_storage": same_storage(tensor, base),
        "same_first_element_pointer": tensor.data_ptr() == base.data_ptr(),
        "values": tensor.tolist(),
    }


def run_experiment() -> dict:
    examples = build_examples()
    base = examples["base"]
    try:
        examples["transpose"].view(-1)
    except RuntimeError:
        incompatible_view_rejected = True
    else:
        incompatible_view_rejected = False

    # detach removes the gradient connection but retains the storage alias.
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    detached = parameter.detach()
    snapshot = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(10)

    return {
        "experiment": "D01 tensor layout (CPU only)",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "experiment_device": "cpu",
            "dtype": "float32",
        },
        "tensors": {name: describe(value, base) for name, value in examples.items()},
        "incompatible_view_rejected": incompatible_view_rejected,
        "snapshot_demo": {
            "parameter_after_update": parameter.tolist(),
            "detach_after_update": detached.tolist(),
            "clone_after_update": snapshot.tolist(),
            "detach_shares_storage": same_storage(parameter, detached),
            "clone_shares_storage": same_storage(parameter, snapshot),
        },
        "limitations": [
            "No GPU timing or performance claim.",
            "No concurrency: clone alone is not an atomic multi-tensor snapshot protocol.",
            "Running the reference experiment does not certify learner understanding.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run_experiment(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
