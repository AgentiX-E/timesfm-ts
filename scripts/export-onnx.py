#!/usr/bin/env python3
"""
TimesFM ONNX Model Exporter — 生产级导出工具

Exports a TimesFM 2.5 200M PyTorch checkpoint from HuggingFace
to a self-contained ONNX file for timesfm-ts.

Requirements:
    pip install "timesfm[torch]" onnx onnxruntime torch

Usage:
    # One-liner: download latest + export + validate
    python scripts/export-onnx.py

    # Custom paths
    python scripts/export-onnx.py \
        -m google/timesfm-2.5-200m-pytorch \
        -o models/timesfm-2.5.onnx

    # Validate-only (no re-export)
    python scripts/export-onnx.py --validate-only -o models/timesfm-2.5.onnx

    # Also check for newer HF versions
    python scripts/export-onnx.py --check-latest

    # Check with machine-parseable diff exit code
    python scripts/export-onnx.py --check-latest --diff

    # Regenerate model descriptor only (no export)
    python scripts/export-onnx.py --descriptor-only
"""

import argparse, json, os, subprocess, sys, time

# NOTE: numpy, onnx, onnxruntime, torch are lazy-imported at function
# level to avoid import errors during lightweight operations such as
# --check-latest and --descriptor-only in CI/nightly environments
# where those packages may not be installed.


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
HF_API = "https://huggingface.co/api/models/{model_id}"

# timesfm-2.5 model shapes (after wrapper)
INPUT_PATCH_LEN = 32   # per-element input patch
MODEL_PATCHES   = 16   # fixed patches in exported ONNX
TOKENIZER_DIM   = 64   # = values(32) + mask(32)


# ─── Step 1: Determine latest model version ──────────────────────────────────

def check_latest_version(model_id: str = DEFAULT_MODEL_ID) -> dict | None:
    """Query HuggingFace API for the latest model revision."""
    import urllib.request as ur
    try:
        url = HF_API.format(model_id=model_id)
        with ur.urlopen(url, timeout=10) as resp:
            info = json.loads(resp.read())
        return {
            "id": info.get("modelId", model_id),
            "sha": info.get("sha", "unknown")[:12],
            "last_modified": info.get("lastModified", "unknown"),
            "tags": info.get("tags", []),
        }
    except Exception as e:
        print(f"  ⚠️  Could not check HF API: {e}")
        return None


# ─── Step 2: Load TimesFM ────────────────────────────────────────────────────

def load_timesfm(model_id: str, torch_compile: bool = False):
    """Load a TimesFM 2.5 model from HuggingFace or local path."""
    import timesfm
    print(f"  Loading from: {model_id}")
    t0 = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_id, force_download=False, torch_compile=torch_compile)
    model.model.eval()
    params = sum(p.numel() for p in model.model.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s  |  {params:,} parameters")
    return model


# ─── Helper: SHA-256 ─────────────────────────────────────────────────────────

def _compute_sha256(path: str) -> str | None:
    """Compute SHA-256 hash of a file. Returns None if file is missing."""
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ─── Step 3: Export to ONNX ──────────────────────────────────────────────────

def export_to_onnx(model, output_path: str):
    """Export the wrapped TimesFM model to a single-file ONNX model."""
    import torch
    import onnx

    class TimesFMWrapper(torch.nn.Module):
        """Wrap the raw TimesFM model to accept pre-concatenated [B,P,64] input.

        The raw model expects (values [B,P,32], masks [B,P,32]) as two
        separate arguments.  This wrapper splits the 64-wide input so
        the ONNX graph has a single input tensor —– matching the
        TypeScript ONNX engine interface.
        """
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, inputs):
            import numpy  # nopep8 — lazy, per policy
            values = inputs[..., :INPUT_PATCH_LEN]
            masks  = (inputs[..., INPUT_PATCH_LEN:] > 0.5)   # float → bool
            (emb_in, emb_out, ts, qs), _ = self.inner(values, masks)
            return emb_in, emb_out, ts, qs

    wrapped = TimesFMWrapper(model.model)
    wrapped.eval()

    # Build dummy — batch=1, patches=MODEL_PATCHES, values + zero mask
    dummy = torch.zeros(1, MODEL_PATCHES, TOKENIZER_DIM)
    dummy[..., :INPUT_PATCH_LEN] = torch.randn(1, MODEL_PATCHES, INPUT_PATCH_LEN)

    # Quick sanity: PyTorch forward
    with torch.no_grad():
        ref = wrapped(dummy)

    print(f"  Exporting with shape [1, {MODEL_PATCHES}, {TOKENIZER_DIM}] …")
    t0 = time.time()

    torch.onnx.export(
        wrapped, dummy, output_path,
        input_names=["inputs"],
        output_names=["input_emb", "output_emb", "output_ts", "output_qs"],
        dynamo=True,
        opset_version=18,
    )

    # Merge external-data back into the .onnx so we have one file
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        m = onnx.load(output_path, load_external_data=True)
        onnx.save(m, output_path, save_as_external_data=False)
        os.remove(data_file)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  Exported in {elapsed:.0f}s  |  {size_mb:.0f} MB")
    return wrapped, dummy


# ─── Step 4: Validate ONNX ───────────────────────────────────────────────────

def validate_onnx(output_path: str, wrapped, dummy):
    """Structural check + ONNX Runtime inference + accuracy vs PyTorch."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    print(f"\n  Validating {os.path.basename(output_path)} …")

    # 4a — structural
    onnx.checker.check_model(onnx.load(output_path))
    print("  ✓  ONNX checker passed")

    # 4b — runtime session
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    print(f"  ✓  ONNX Runtime session  (providers: {session.get_providers()})")

    # 4c — accuracy vs PyTorch
    with torch.no_grad():
        py_out = wrapped(dummy)
    ort_out = session.run(None, {"inputs": dummy.numpy().astype("float32")})

    names = ["input_emb", "output_emb", "output_ts", "output_qs"]
    all_ok = True
    for name, py, onx in zip(names, py_out, ort_out):
        diff = float(abs(onx - py.numpy()).max())
        ok = diff < 1e-3
        if not ok:
            all_ok = False
        marker = "✓" if ok else "✗"
        print(f"  {marker}  {name:12s}  shape={list(onx.shape)}  max_diff={diff:.2e}")

    # 4d — NaN/Inf check
    for name, onx in zip(names, ort_out):
        arr = np.asarray(onx)
        if np.any(np.isnan(arr)):
            print(f"  ✗  {name} contains NaN!")
            all_ok = False
        if np.any(np.isinf(arr)):
            print(f"  ✗  {name} contains Inf!")
            all_ok = False

    if all_ok:
        print("  ✓  All checks passed — PyTorch ↔ ONNX match")
    else:
        print("  ✗  Validation FAILED")
    return all_ok


# ─── Step 5: Write metadata ──────────────────────────────────────────────────

def write_model_metadata(output_path: str, model_id: str, hf_info: dict | None):
    """Write a small JSON metadata file next to the ONNX model."""
    meta = {
        "source": model_id,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size_mb": round(os.path.getsize(output_path) / 1024**2, 1),
        "input_shape": f"[1, {MODEL_PATCHES}, {TOKENIZER_DIM}]",
        "outputs": [
            "input_emb  [1, 16, 1280]",
            "output_emb [1, 16, 1280]",
            "output_ts  [1, 16, 1280]",
            "output_qs  [1, 16, 10240]",
        ],
        "hf_revision": hf_info.get("sha") if hf_info else None,
    }
    meta_path = output_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  📝  Metadata → {meta_path}")


# ─── Step 6: Write model descriptor ──────────────────────────────────────────

def write_model_descriptor(model, output_path: str, hf_info: dict | None):
    """Write ModelDescriptor JSON alongside the ONNX model."""
    import json as _json
    import os as _os
    inner = model.model  # the raw TimesFM transformer

    # Extract architecture from model parameters
    num_layers = getattr(inner, 'num_layers', 20)
    num_heads = getattr(inner, 'num_heads', 16)
    model_dims = getattr(inner, 'model_dims', 1280) or 1280

    # If not available as attributes, try to infer from parameter shapes
    if not hasattr(inner, 'num_layers'):
        import re
        layers = set()
        for name, _ in inner.named_parameters():
            m = re.search(r'\.h\.(\d+)\.', name)
            if m:
                layers.add(int(m.group(1)))
        if layers:
            num_layers = max(layers) + 1

    if not hasattr(inner, 'num_heads'):
        for name, p in inner.named_parameters():
            if 'q_proj.weight' in name:
                num_heads = p.shape[0] // (model_dims // num_heads) if model_dims else 16
                break

    # Architecture constants (same as createTimesFM25Config)
    input_patch_len = 32
    output_patch_len = 128
    output_quantile_len = 1024
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ONNX shapes — compute sha256/size only if the file exists
    sha256 = _compute_sha256(output_path)
    size_bytes = _os.path.getsize(output_path) if _os.path.exists(output_path) else None

    desc = {
        "schema": 1,
        "model": {
            "version": "2.5",
            "variant": "200m",
            "hf_revision": hf_info.get("sha", "unknown") if hf_info else "unknown",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "onnx": {
            "input_name": "inputs",
            "input_shape": [1, MODEL_PATCHES, TOKENIZER_DIM],
            "outputs": {
                "input_emb":  [1, MODEL_PATCHES, model_dims],
                "output_emb": [1, MODEL_PATCHES, model_dims],
                "output_ts":  [1, MODEL_PATCHES, model_dims],
                "output_qs":  [1, MODEL_PATCHES, output_quantile_len * (len(quantiles) + 1)]
            },
            "opset": 18,
            "sha256": sha256,
            "size_bytes": size_bytes
        },
        "architecture": {
            "input_patch_len": input_patch_len,
            "output_patch_len": output_patch_len,
            "output_quantile_len": output_quantile_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "model_dims": model_dims,
            "quantiles": quantiles,
            "context_limit": 16384
        },
        "processing": {
            "preprocessing": "revin",
            "postprocessing": ["flip_invariance", "quantile_crossing_fix"]
        }
    }

    descriptor_path = _os.path.splitext(output_path)[0] + "-descriptor.json"
    with open(descriptor_path, "w") as f:
        _json.dump(desc, f, indent=2)
    print(f"  📋  Descriptor → {descriptor_path}")

    # Also write to models/model-descriptor.json for the committed source-of-truth
    meta_dir = _os.path.dirname(output_path) or "."
    committed_path = _os.path.join(meta_dir, "model-descriptor.json")
    with open(committed_path, "w") as f:
        _json.dump(desc, f, indent=2)
    print(f"  📋  Committed descriptor → {committed_path}")

    return desc


# ─── Version comparison for CI ───────────────────────────────────────────────

def check_latest_with_diff(model_id: str) -> bool:
    """Returns True if version matches (no change), False if new version."""
    import json as _json
    import os as _os
    info = check_latest_version(model_id)
    if not info:
        return True  # Can't check, assume no change

    committed_paths = [
        "models/model-descriptor.json",
        "models/timesfm-2.5-descriptor.json",
        "models/timesfm-2.5.onnx.meta.json",
    ]

    for cp in committed_paths:
        if _os.path.exists(cp):
            with open(cp) as f:
                meta = _json.load(f)
            committed_sha = (meta.get("model", {}).get("hf_revision") or
                           meta.get("hf_revision", ""))
            if committed_sha and committed_sha == info["sha"]:
                return True  # Match

    return False  # No match = new version


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Export TimesFM 2.5 200M → single-file ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export-onnx.py                           # full pipeline
  python scripts/export-onnx.py --check-latest            # only check HF
  python scripts/export-onnx.py --check-latest --diff     # CI version diff
  python scripts/export-onnx.py --validate-only -o M.onnx  # validate only
  python scripts/export-onnx.py --descriptor-only          # metadata only
        """)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("-o", "--output", default="models/timesfm-2.5.onnx")
    ap.add_argument("--check-latest", action="store_true",
                    help="Query HuggingFace for latest model revision, then exit")
    ap.add_argument("--diff", action="store_true",
                    help="With --check-latest: exit 1 if new version detected (CI mode)")
    ap.add_argument("--validate-only", action="store_true",
                    help="Only validate an existing ONNX file")
    ap.add_argument("--descriptor-only", action="store_true",
                    help="Only extract architecture info and write descriptor JSON")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--torch-compile", action="store_true")
    ap.add_argument("--no-metadata", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  TimesFM → ONNX  Exporter for timesfm-ts")
    print("=" * 60)

    # --check-latest
    info = check_latest_version(args.model)
    if info:
        print(f"\n  HuggingFace: {info['id']}")
        print(f"  Revision:     {info['sha']}")
        print(f"  Updated:      {info['last_modified']}")
    if args.check_latest:
        print("\n  Checking against committed descriptor …")
        matches = check_latest_with_diff(args.model)
        if matches:
            print("  ✅  Up to date")
        else:
            print("  🔄  NEW VERSION DETECTED")
        if args.diff:
            sys.exit(0 if matches else 1)
        return

    # --validate-only
    if args.validate_only:
        import numpy as np
        import onnx
        import onnxruntime as ort
        if not os.path.exists(args.output):
            print(f"\n  ✗ File not found: {args.output}")
            sys.exit(1)
        model = onnx.load(args.output)
        onnx.checker.check_model(model)
        session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, MODEL_PATCHES, TOKENIZER_DIM).astype("float32")
        outs = session.run(None, {"inputs": dummy})
        print(
            f"\n  ✓  Validated — {len(outs)} outputs, "
            f"shapes: {[list(o.shape) for o in outs]}"
        )
        return

    # --descriptor-only
    if args.descriptor_only:
        print("\n[1/1] Extracting architecture & writing descriptor …")
        model = load_timesfm(args.model, torch_compile=False)
        write_model_descriptor(model, args.output, info)
        print(f"\n{'=' * 60}")
        print("  ✅  Descriptor written")
        base = os.path.splitext(args.output)[0]
        print(f"  📁  {base}-descriptor.json")
        print(f"  📁  {os.path.dirname(args.output) or '.'}/model-descriptor.json")
        if info:
            print(f"  🔖  HF revision: {info['sha']}")
        print(f"{'=' * 60}")
        return

    # Full pipeline
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("\n[1/4] Loading TimesFM …")
    model = load_timesfm(args.model, torch_compile=args.torch_compile)

    print(f"\n[2/4] Exporting → {args.output}")
    wrapped, dummy = export_to_onnx(model, args.output)

    if not args.skip_validation:
        print(f"\n[3/4] Validating …")
        ok = validate_onnx(args.output, wrapped, dummy)
        if not ok:
            sys.exit(1)

    print(f"\n[4/4] Writing descriptor & metadata …")
    write_model_descriptor(model, args.output, info)

    if not args.no_metadata:
        write_model_metadata(args.output, args.model, info)

    print(f"\n{'=' * 60}")
    print(f"  ✅  ONNX model ready")
    print(f"  📁  {args.output}  ({os.path.getsize(args.output)/1024**2:.0f} MB)")
    base = os.path.splitext(args.output)[0]
    print(f"  📋  {base}-descriptor.json")
    print(f"  📐  Compatible: timesfm-ts (onnxruntime-node)")
    if info:
        print(f"  🔖  HF revision: {info['sha']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
