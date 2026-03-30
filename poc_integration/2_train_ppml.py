from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    dataset_path = root / "encrypted_dataset.json"
    metadata_path = root / "dataset_metadata.json"
    model_export_path = root / "model_export.json"

    try:
        import blindml
    except ImportError as exc:
        print(
            "Failed to import blindml. Build/install the Python bindings first with "
            "`maturin develop --release -m blindml/Cargo.toml`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    blindml.load_and_validate_package(str(dataset_path), str(metadata_path))
    blindml.export_poc_model(
        str(dataset_path),
        str(metadata_path),
        str(model_export_path),
    )

    print("PPML PoC validation succeeded.")
    print(f"Model export written to {model_export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
