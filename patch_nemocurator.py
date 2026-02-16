#!/usr/bin/env python3
"""
Patch NeMo Curator v1.0 source files for nvidia/llama-embed-nemotron-8b compatibility.

NeMo Curator v1.0 has several issues with custom HuggingFace models:
  1. AutoModel.from_pretrained() called without trust_remote_code=True
  2. AutoConfig / AutoTokenizer called without trust_remote_code=True
  3. Model loads in float32 instead of bfloat16 (doubles GPU memory)
  4. Embeddings written as float64 instead of float32 (wasted disk space)

Since Ray workers import the package fresh in each actor process,
monkey-patching in the main process has no effect. The installed source
files must be patched directly.

USAGE:
    python patch_nemocurator.py          # apply patches
    python patch_nemocurator.py --check  # check-only (exit 0 if patched, 1 if not)

Can also be imported and called programmatically:
    from patch_nemocurator import ensure_patches
    ensure_patches()  # idempotent
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Tuple


def _get_nemo_curator_dir() -> Path:
    """Locate the nemo_curator package directory."""
    try:
        import nemo_curator
        return Path(nemo_curator.__file__).parent
    except ImportError:
        print("ERROR: nemo_curator is not installed in this environment.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Patch definitions: (file_relative_path, old_string, new_string, description)
# ---------------------------------------------------------------------------

def _get_patches(site: Path) -> List[Tuple[Path, str, str, str]]:
    """Return list of (filepath, old, new, description) patches."""
    base_py = site / "stages" / "text" / "embedders" / "base.py"
    tokenizer_py = site / "stages" / "text" / "models" / "tokenizer.py"

    return [
        # --- base.py: EmbeddingModelStage.setup ---
        (
            base_py,
            (
                "self.model = AutoModel.from_pretrained("
                "self.model_identifier, local_files_only=True)"
            ),
            (
                "self.model = AutoModel.from_pretrained(\n"
                "            self.model_identifier,\n"
                "            local_files_only=True,\n"
                "            trust_remote_code=True,\n"
                "            torch_dtype=torch.bfloat16,\n"
                '            attn_implementation="eager",\n'
                "        )"
            ),
            "EmbeddingModelStage.setup: trust_remote_code + bfloat16 + eager attention",
        ),
        # --- base.py: collect_outputs – float32 instead of float64 ---
        (
            base_py,
            "return torch.cat(processed_outputs, dim=0).numpy().tolist()",
            "return torch.cat(processed_outputs, dim=0).float().numpy()  # [N, dim] float32",
            "collect_outputs: float32 numpy instead of float64 list",
        ),
        # --- base.py: create_output_dataframe – split 2D array into list of 1D ---
        (
            base_py,
            "return df_cpu.assign(**{self.embedding_field: collected_output})",
            (
                "embeddings_list = [collected_output[i] for i in range(len(collected_output))]\n"
                "        return df_cpu.assign(**{self.embedding_field: embeddings_list})"
            ),
            "create_output_dataframe: split 2D array into list of 1D float32 arrays",
        ),
        # --- tokenizer.py: load_cfg – trust_remote_code ---
        (
            tokenizer_py,
            (
                "return AutoConfig.from_pretrained(\n"
                "            self.model_identifier, cache_dir=self.cache_dir, "
                "local_files_only=local_files_only\n"
                "        )"
            ),
            (
                "return AutoConfig.from_pretrained(\n"
                "            self.model_identifier, cache_dir=self.cache_dir, "
                "local_files_only=local_files_only, trust_remote_code=True\n"
                "        )"
            ),
            "TokenizerStage.load_cfg: trust_remote_code=True",
        ),
        # --- tokenizer.py: _setup – trust_remote_code ---
        (
            tokenizer_py,
            (
                "self.tokenizer = AutoTokenizer.from_pretrained(\n"
                "            self.model_identifier,\n"
                "            padding_side=self.padding_side,\n"
                "            cache_dir=self.cache_dir,\n"
                "            local_files_only=local_files_only,\n"
                "        )"
            ),
            (
                "self.tokenizer = AutoTokenizer.from_pretrained(\n"
                "            self.model_identifier,\n"
                "            padding_side=self.padding_side,\n"
                "            cache_dir=self.cache_dir,\n"
                "            local_files_only=local_files_only,\n"
                "            trust_remote_code=True,\n"
                "        )"
            ),
            "TokenizerStage._setup: trust_remote_code=True",
        ),
    ]


def check_patches(site: Path) -> Tuple[List[str], List[str]]:
    """
    Check which patches are applied and which are pending.

    Returns (applied, pending) lists of patch descriptions.
    """
    applied: List[str] = []
    pending: List[str] = []

    for filepath, old_str, new_str, desc in _get_patches(site):
        if not filepath.exists():
            pending.append(f"{desc} (FILE NOT FOUND: {filepath})")
            continue

        content = filepath.read_text(encoding="utf-8")
        if new_str in content:
            applied.append(desc)
        elif old_str in content:
            pending.append(desc)
        else:
            applied.append(f"{desc} (already modified, pattern not found)")

    return applied, pending


def apply_patches(site: Path, verbose: bool = True) -> int:
    """
    Apply all patches. Returns number of patches applied.
    """
    count = 0
    for filepath, old_str, new_str, desc in _get_patches(site):
        if not filepath.exists():
            if verbose:
                print(f"  SKIP  {desc} — file not found: {filepath}")
            continue

        content = filepath.read_text(encoding="utf-8")

        if new_str in content:
            if verbose:
                print(f"  OK    {desc} — already patched")
            continue

        if old_str not in content:
            if verbose:
                print(f"  WARN  {desc} — original pattern not found (manual edit or different version?)")
            continue

        content = content.replace(old_str, new_str, 1)
        filepath.write_text(content, encoding="utf-8")
        count += 1
        if verbose:
            print(f"  PATCH {desc}")

    return count


def ensure_patches(verbose: bool = False) -> None:
    """Idempotent entry point: apply patches if needed, skip if already done."""
    site = _get_nemo_curator_dir()
    _, pending = check_patches(site)
    if pending:
        if verbose:
            print(f"NeMo Curator patches: {len(pending)} pending, applying ...")
        applied = apply_patches(site, verbose=verbose)
        if verbose:
            print(f"NeMo Curator patches: {applied} applied")
    elif verbose:
        print("NeMo Curator patches: all applied")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch NeMo Curator for nvidia/llama-embed-nemotron-8b",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check patch status only (exit 0 = all patched, exit 1 = pending patches)",
    )
    args = parser.parse_args()

    site = _get_nemo_curator_dir()
    print(f"NeMo Curator location: {site}")

    applied, pending = check_patches(site)

    if args.check:
        print(f"\nApplied ({len(applied)}):")
        for desc in applied:
            print(f"  [OK] {desc}")
        if pending:
            print(f"\nPending ({len(pending)}):")
            for desc in pending:
                print(f"  [!!] {desc}")
            sys.exit(1)
        else:
            print("\nAll patches applied.")
            sys.exit(0)

    # Apply mode
    if not pending:
        print("All patches already applied — nothing to do.")
        return

    print(f"\n{len(pending)} patch(es) to apply:\n")
    count = apply_patches(site, verbose=True)
    print(f"\nDone: {count} patch(es) applied.")


if __name__ == "__main__":
    main()
