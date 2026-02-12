#!/usr/bin/env python3
"""
Thin Python driver for manifest generation.

Loads chutes-bcm.so via ctypes and uses it to:
1. Validate CFSV_OP (internal to C, decision not exposed)
2. Walk directories, compile .py files
3. Pass code objects to C for hashing (real or fake, decided by C)
4. Write binary manifest via C

The Python code never sees whether hashes are real or fake.

Usage:
    CFSV_OP=$SECRET python3 generate_manifest_driver.py \
        --output /etc/bytecode.manifest \
        --json-output /tmp/bytecode.manifest.json \
        --lib /tmp/chutes-bcm.so \
        [--extra-dirs /app,/custom]
"""

import argparse
import ctypes
import json
import marshal
import os
import struct
import sys
import types


def load_library(lib_path=None):
    """Load chutes-bcm.so from an explicit path or standard search paths."""
    if lib_path and os.path.exists(lib_path):
        return ctypes.PyDLL(lib_path)
    search_paths = [
        "/usr/local/lib/chutes-bcm.so",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "chutes-bcm.so"),
        "./chutes-bcm.so",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return ctypes.PyDLL(path)
    raise RuntimeError("chutes-bcm.so not found")


def compile_py_file(filepath):
    """Compile a .py file and return its code object, or None on failure."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        return compile(source, filepath, "exec")
    except (SyntaxError, ValueError, TypeError, UnicodeDecodeError):
        return None
    except Exception:
        return None


def load_pyc_code(filepath):
    """Load a .pyc file and return its code object, or None on failure."""
    try:
        with open(filepath, "rb") as f:
            f.read(4)  # magic number
            flags = struct.unpack("<I", f.read(4))[0]
            if flags & 0x1:
                f.read(8)  # hash-based: skip hash
            else:
                f.read(8)  # timestamp-based: skip timestamp + size
            code = marshal.loads(f.read())
        if not isinstance(code, types.CodeType):
            return None
        return code
    except Exception:
        return None


def collect_py_files(search_dirs):
    """Walk directories and collect all .py file paths."""
    seen = set()
    files = []

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, filenames in os.walk(search_dir, followlinks=True):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.realpath(os.path.join(root, fname))
                if filepath in seen:
                    continue
                seen.add(filepath)
                files.append(filepath)

    return files


def collect_pyc_only_files(search_dirs, already_seen):
    """Collect .pyc files that have no corresponding .py source."""
    pyc_entries = []  # list of (py_path, pyc_path)

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, filenames in os.walk(search_dir, followlinks=True):
            for fname in filenames:
                if not fname.endswith(".pyc"):
                    continue
                pyc_path = os.path.join(root, fname)
                # Determine corresponding .py path
                if "__pycache__" in root:
                    parent = os.path.dirname(root)
                    base = fname.split(".")[0]
                    py_path = os.path.realpath(os.path.join(parent, base + ".py"))
                else:
                    py_path = os.path.realpath(pyc_path[:-1])

                if py_path not in already_seen:
                    already_seen.add(py_path)
                    pyc_entries.append((py_path, pyc_path))

    return pyc_entries


def main():
    parser = argparse.ArgumentParser(description="Generate bytecode manifest")
    parser.add_argument(
        "--output",
        default="/etc/bytecode.manifest",
        help="Output manifest path",
    )
    parser.add_argument(
        "--extra-dirs",
        default="",
        help="Comma-separated extra directories to scan",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Also write cleartext JSON manifest (path -> hash_hex)",
    )
    parser.add_argument(
        "--lib",
        default=None,
        help="Path to chutes-bcm.so (overrides default search)",
    )
    args = parser.parse_args()

    # Load the C library
    lib = load_library(args.lib)

    # Set up function signatures
    lib.manifest_init.argtypes = []
    lib.manifest_init.restype = ctypes.c_int

    lib.manifest_hash_code.argtypes = [
        ctypes.py_object,  # PyObject *code
        ctypes.c_char_p,  # const char *filepath
        ctypes.POINTER(ctypes.c_uint8 * 32),  # uint8_t hash_out[32] (HMAC-SHA256)
    ]
    lib.manifest_hash_code.restype = ctypes.c_int

    lib.manifest_write.argtypes = [
        ctypes.c_char_p,  # const char *output_path
        ctypes.POINTER(ctypes.c_char_p),  # const char **paths
        ctypes.POINTER(ctypes.c_uint8),  # const uint8_t *hashes
        ctypes.c_uint32,  # uint32_t count
    ]
    lib.manifest_write.restype = ctypes.c_int

    # Initialize (checks CFSV_OP internally)
    lib.manifest_init()

    # Build search directory list
    search_dirs = []

    for p in sys.path:
        if p and os.path.isdir(p):
            rp = os.path.realpath(p)
            if rp not in search_dirs:
                search_dirs.append(rp)

    for extra in ["/app", "/usr/lib", "/usr/local/lib"]:
        rp = os.path.realpath(extra)
        if os.path.isdir(rp) and rp not in search_dirs:
            search_dirs.append(rp)

    if args.extra_dirs:
        for d in args.extra_dirs.split(","):
            d = d.strip()
            if d and os.path.isdir(d):
                rp = os.path.realpath(d)
                if rp not in search_dirs:
                    search_dirs.append(rp)

    print(f"[manifest] Scanning {len(search_dirs)} directories...")

    # Collect all .py files
    py_files = collect_py_files(search_dirs)
    print(f"[manifest] Found {len(py_files)} .py files")

    # Compile and hash each file
    entries = []  # list of (path, 32-byte HMAC-SHA256 hash)
    hash_buf = (ctypes.c_uint8 * 32)()
    seen_paths = set(py_files)

    for filepath in py_files:
        code = compile_py_file(filepath)
        if code is None:
            continue

        ret = lib.manifest_hash_code(
            code,
            filepath.encode("utf-8"),
            ctypes.byref(hash_buf),
        )
        if ret == 0:
            entries.append((filepath, bytes(hash_buf)))

    # Second pass: .pyc-only files (no corresponding .py source)
    pyc_files = collect_pyc_only_files(search_dirs, seen_paths)
    print(f"[manifest] Found {len(pyc_files)} .pyc-only files")

    for py_path, pyc_path in pyc_files:
        code = load_pyc_code(pyc_path)
        if code is None:
            continue

        ret = lib.manifest_hash_code(
            code,
            py_path.encode("utf-8"),
            ctypes.byref(hash_buf),
        )
        if ret == 0:
            entries.append((py_path, bytes(hash_buf)))

    # Sort by path for binary search at runtime
    entries.sort(key=lambda x: x[0])

    print(f"[manifest] Hashed {len(entries)} files successfully")

    # Write cleartext JSON manifest if requested (for validator).
    if args.json_output:
        json_manifest = {
            "entries": {path: {"hash_hex": hash_bytes.hex()} for path, hash_bytes in entries}
        }
        with open(args.json_output, "w") as f:
            json.dump(json_manifest, f)
        print(f"[manifest] Wrote JSON index ({len(entries)} entries) to {args.json_output}")

    # Prepare arrays for C
    count = len(entries)
    paths_array = (ctypes.c_char_p * count)()
    hashes_array = (ctypes.c_uint8 * (count * 32))()

    for i, (path, hash_bytes) in enumerate(entries):
        paths_array[i] = path.encode("utf-8")
        for j in range(32):
            hashes_array[i * 32 + j] = hash_bytes[j]

    # Write manifest via C
    ret = lib.manifest_write(
        args.output.encode("utf-8"),
        paths_array,
        hashes_array,
        count,
    )

    if ret != 0:
        err_map = {
            -1: "invalid arguments",
            -2: f"cannot open {args.output} (permission denied?)",
            -3: "header write failed",
            -4: "entry write failed",
        }
        print(f"[manifest] ERROR: manifest_write returned {ret}: {err_map.get(ret, 'unknown')}")
        sys.exit(1)

    file_size = os.path.getsize(args.output)
    print(f"[manifest] Wrote {count} entries ({file_size} bytes) to {args.output}")


if __name__ == "__main__":
    main()
