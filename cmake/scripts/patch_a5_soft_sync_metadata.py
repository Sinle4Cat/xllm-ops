#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disable the FFTS-address requirement for validated A5 soft-sync kernels."""

import argparse
import json
import os
from pathlib import Path
import tempfile


def patch_metadata(path: Path) -> None:
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("coreType") != "MIX":
        raise ValueError(f"{path}: expected coreType=MIX")
    if metadata.get("intercoreSync") != 1:
        raise ValueError(f"{path}: expected intercoreSync=1 before patching")

    kernels = metadata.get("kernelList")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError(f"{path}: missing kernelList")
    for kernel in kernels:
        if kernel.get("taskRation") != "1:2":
            raise ValueError(f"{path}: expected only MIX 1:2 kernels")
        if kernel.get("crossCoreSync") != 1:
            raise ValueError(
                f"{path}: expected crossCoreSync=1 before patching"
            )

    # The 1:2 scheduling contract is retained.  Only the runtime request for
    # an FFTS hardware-sync address is removed; synchronization is implemented
    # by PTO's A5 MIX software rendezvous with GM generation counters.
    metadata["intercoreSync"] = 0
    for kernel in kernels:
        kernel["crossCoreSync"] = 0

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)
            file.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata_files = sorted(args.kernel_dir.glob("*.json"))
    if not metadata_files:
        raise FileNotFoundError(
            f"no generated kernel metadata under {args.kernel_dir}"
        )
    for path in metadata_files:
        patch_metadata(path)


if __name__ == "__main__":
    main()
