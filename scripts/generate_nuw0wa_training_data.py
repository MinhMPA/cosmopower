#!/usr/bin/env python3
"""Build paper-style CosmoPower training data for a CAMB-based nuw0waCDM model.

This script follows the Section 3 package ideas from the 2025 CosmoPower paper,
while remaining usable with the repository layout currently visible in
cosmopower/training.

Subcommands
-----------
init
    Create the Latin-hypercube parameter table and pre-allocate HDF5 quantity files.
run-shard
    Compute one shard of spectra/background quantities and fill the matching HDF5 files.
run-all
    Convenience wrapper: init if needed, then run every shard sequentially.
status
    Report how many samples have been filled per quantity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.stats import qmc


STR_DTYPE = h5py.string_dtype(encoding="utf-8")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration {path} did not parse to a mapping.")
    return config


def parse_quantity_arg(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    out = [item.strip() for item in text.split(",") if item.strip()]
    return out or None


def resolve_outdir(config: Mapping[str, Any], outdir: Optional[str]) -> Path:
    if outdir:
        return Path(outdir)
    default_name = str(config.get("path") or config.get("network_name") or "cosmopower_dataset")
    return Path(default_name)


def get_generator_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    gen = config.get("generator", {})
    if gen is None:
        gen = {}
    if not isinstance(gen, dict):
        raise ValueError("generator block must be a mapping")
    return gen


def quantity_to_stem(quantity: str) -> str:
    return quantity.replace("/", "_").replace("-", "_")


def split_parameter_blocks(parameters: Mapping[str, Any]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str], Dict[str, float], List[str]]:
    sampled: Dict[str, Tuple[float, float]] = {}
    derived: Dict[str, str] = {}
    fixed: Dict[str, float] = {}
    computed: List[str] = []

    for name, value in parameters.items():
        if isinstance(value, list) and len(value) == 2:
            sampled[name] = (float(value[0]), float(value[1]))
        elif isinstance(value, str) and value.strip().startswith("lambda"):
            derived[name] = value.strip()
        elif value is None:
            computed.append(name)
        else:
            fixed[name] = float(value)

    return sampled, derived, fixed, computed


def make_lhs(sampled: Mapping[str, Tuple[float, float]], n_samples: int, seed: int) -> Tuple[List[str], np.ndarray]:
    names = list(sampled.keys())
    if not names:
        raise ValueError("No sampled parameters found in configuration.")
    lows = np.array([sampled[name][0] for name in names], dtype=np.float64)
    highs = np.array([sampled[name][1] for name in names], dtype=np.float64)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=n_samples)
    scaled = qmc.scale(unit, lows, highs)
    return names, scaled.T  # p x N


def evaluate_lambda_vector(expr: str, context: Mapping[str, np.ndarray], n_samples: int) -> np.ndarray:
    func = eval(expr, {"np": np, "math": math}, {})
    signature = inspect.signature(func)
    kwargs = {name: context[name] for name in signature.parameters}
    out = np.asarray(func(**kwargs), dtype=np.float64)
    if out.shape == ():
        out = np.full(n_samples, float(out), dtype=np.float64)
    if out.shape != (n_samples,):
        raise ValueError(f"Derived expression {expr!r} returned shape {out.shape}, expected ({n_samples},)")
    return out


def evaluate_lambda_scalar(expr: str, context: Mapping[str, float]) -> float:
    func = eval(expr, {"np": np, "math": math}, {})
    signature = inspect.signature(func)
    kwargs = {name: context[name] for name in signature.parameters}
    return float(func(**kwargs))


def build_parameter_tables(config: Mapping[str, Any], n_samples: int, seed: int) -> Dict[str, Any]:
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters block must be present and must be a mapping")

    sampled, derived, fixed, computed = split_parameter_blocks(parameters)
    sampled_names, sampled_table = make_lhs(sampled, n_samples=n_samples, seed=seed)

    vector_context: Dict[str, np.ndarray] = {
        name: sampled_table[i].astype(np.float64, copy=False) for i, name in enumerate(sampled_names)
    }
    for name, value in fixed.items():
        vector_context[name] = np.full(n_samples, value, dtype=np.float64)

    derived_names = list(derived.keys())
    derived_arrays: Dict[str, np.ndarray] = {}
    for name, expr in derived.items():
        derived_arrays[name] = evaluate_lambda_vector(expr, vector_context, n_samples)
        vector_context[name] = derived_arrays[name]

    emulated_code = config.get("emulated_code", {})
    full_input_names = list(emulated_code.get("inputs", []))
    if not full_input_names:
        raise ValueError("emulated_code.inputs must be specified")
    missing = [name for name in full_input_names if name not in vector_context]
    if missing:
        raise ValueError(f"emulated_code.inputs contains names that are not sampled/fixed/derived: {missing}")

    full_input_table = np.vstack([vector_context[name] for name in full_input_names]).astype(np.float64)
    derived_table = (
        np.vstack([derived_arrays[name] for name in derived_names]).astype(np.float64)
        if derived_names
        else np.zeros((0, n_samples), dtype=np.float64)
    )
    fixed_names = list(fixed.keys())
    fixed_values = np.array([fixed[name] for name in fixed_names], dtype=np.float64)

    return {
        "sampled_names": sampled_names,
        "sampled_table": sampled_table.astype(np.float64),
        "derived_names": derived_names,
        "derived_table": derived_table,
        "fixed_names": fixed_names,
        "fixed_values": fixed_values,
        "computed_names": computed,
        "full_input_names": full_input_names,
        "full_input_table": full_input_table,
    }


def build_shard_ranges(n_samples: int, n_shards: int) -> List[Tuple[int, int]]:
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    if n_shards > n_samples:
        raise ValueError("n_shards cannot exceed Ntraining")
    edges = np.linspace(0, n_samples, n_shards + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_shards)]


def infer_created_quantities(config: Mapping[str, Any]) -> List[str]:
    networks = config.get("networks")
    if not isinstance(networks, list) or not networks:
        raise ValueError("networks block must be a non-empty list")
    out: List[str] = []
    seen = set()
    for net in networks:
        quantity = str(net.get("quantity"))
        if quantity in seen:
            raise ValueError(f"Duplicate network entry for quantity {quantity}")
        seen.add(quantity)
        out.append(quantity)
    return out


def select_networks(config: Mapping[str, Any], quantities: Optional[Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    networks = config.get("networks")
    if not isinstance(networks, list) or not networks:
        raise ValueError("networks block must be a non-empty list")

    selected_set = set(quantities) if quantities else None
    out: Dict[str, Dict[str, Any]] = {}
    for raw in networks:
        quantity = str(raw.get("quantity"))
        if selected_set is not None and quantity not in selected_set:
            continue
        if quantity in out:
            raise ValueError(f"Duplicate network entry for quantity {quantity}")
        net = dict(raw)
        inputs = list(net.get("inputs") or [])
        if quantity != "derived" and not inputs:
            raise ValueError(f"Network for {quantity} must define inputs explicitly in this script")
        if quantity == "derived" and not inputs:
            raise ValueError("derived network must define inputs explicitly in this script")
        net["inputs"] = inputs
        net["stem"] = quantity_to_stem(quantity)
        net["modes_array"] = make_mode_array(net)
        out[quantity] = net

    if selected_set is not None:
        missing = sorted(selected_set - set(out.keys()))
        if missing:
            raise ValueError(f"Requested quantities are not defined in the config: {missing}")
    if not out:
        raise ValueError("No quantities selected")
    return out


def make_mode_array(network: Mapping[str, Any]) -> np.ndarray:
    quantity = str(network["quantity"])
    if quantity == "derived":
        outputs = network.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("derived network requires a non-empty outputs list")
        return np.asarray([str(item) for item in outputs], dtype=object)

    modes = network.get("modes")
    if not isinstance(modes, dict):
        raise ValueError(f"Network for {quantity} requires a modes block")
    if "range" not in modes:
        raise ValueError(f"Network for {quantity} requires modes.range")
    lo, hi = modes["range"]
    label = str(modes.get("label", ""))

    if quantity.startswith("Cl/"):
        lo_i = int(lo)
        hi_i = int(hi)
        return np.arange(lo_i, hi_i + 1, dtype=np.int32)

    n_modes = int(modes.get("n_modes", 0))
    if n_modes <= 0:
        raise ValueError(f"Network for {quantity} requires a positive modes.n_modes")

    if label.lower().startswith("k"):
        return np.logspace(np.log10(float(lo)), np.log10(float(hi)), n_modes, dtype=np.float64)
    if label.lower().startswith("z"):
        return np.linspace(float(lo), float(hi), n_modes, dtype=np.float64)

    raise ValueError(f"Do not know how to build modes for quantity {quantity} with label {label!r}")


def write_string_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    arr = np.asarray(list(values), dtype=object)
    group.create_dataset(name, data=arr, dtype=STR_DTYPE)


def write_metadata_file(
    outdir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    n_samples: int,
    n_shards: int,
    created_quantities: Sequence[str],
    seed: int,
    dtype_name: str,
) -> None:
    payload = {
        "config_file": str(config_path),
        "network_name": str(config.get("network_name", "")),
        "dataset_path": str(outdir),
        "n_samples": int(n_samples),
        "n_shards": int(n_shards),
        "seed": int(seed),
        "dtype": dtype_name,
        "created_quantities": list(created_quantities),
    }
    with (outdir / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_metadata_file(outdir: Path) -> Dict[str, Any]:
    path = outdir / "dataset_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def spectra_dtype_from_name(name: str) -> np.dtype:
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "f4": np.float32,
        "f8": np.float64,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported generator.dtype {name!r}; use float32 or float64")
    return np.dtype(mapping[key])


def initialize_dataset(
    config_path: Path,
    config: Mapping[str, Any],
    outdir: Path,
    quantities: Optional[Sequence[str]],
    n_samples: Optional[int],
    n_shards: Optional[int],
    overwrite: bool,
) -> None:
    gen = get_generator_config(config)
    seed = int(gen.get("seed", 0))
    n_samples_final = int(n_samples or config.get("samples", {}).get("Ntraining", 0))
    n_shards_final = int(n_shards or gen.get("n_shards", 10))
    dtype_name = str(gen.get("dtype", "float32"))
    compression = gen.get("compression", "gzip")
    spectra_dtype = spectra_dtype_from_name(dtype_name)

    if outdir.exists() and overwrite:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_copy = outdir / "config_used.yaml"
    config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    selected_networks = select_networks(config, quantities)
    selected_quantities = list(selected_networks.keys())
    shard_ranges = build_shard_ranges(n_samples_final, n_shards_final)
    tables = build_parameter_tables(config, n_samples=n_samples_final, seed=seed)

    parameters_path = outdir / "parameters.hdf5"
    with h5py.File(parameters_path, "w") as handle:
        handle.attrs["network_name"] = str(config.get("network_name", ""))
        handle.attrs["n_samples"] = int(n_samples_final)
        handle.attrs["seed"] = int(seed)
        handle.attrs["format"] = "cosmopower-paper-style-v1"

        write_string_dataset(handle, "sampled_parameter_names", tables["sampled_names"])
        handle.create_dataset("sampled_parameters", data=tables["sampled_table"], compression=compression)

        write_string_dataset(handle, "derived_parameter_names", tables["derived_names"])
        handle.create_dataset("derived_parameters", data=tables["derived_table"], compression=compression)

        write_string_dataset(handle, "fixed_parameter_names", tables["fixed_names"])
        handle.create_dataset("fixed_parameter_values", data=tables["fixed_values"])

        write_string_dataset(handle, "computed_parameter_names", tables["computed_names"])

        write_string_dataset(handle, "boltzmann_input_names", tables["full_input_names"])
        handle.create_dataset("boltzmann_inputs", data=tables["full_input_table"], compression=compression)

    for quantity, spec in selected_networks.items():
        modes = spec["modes_array"]
        mode_label = "derived_name" if quantity == "derived" else str(spec.get("modes", {}).get("label", ""))
        for shard_id, (start, stop) in enumerate(shard_ranges):
            local_n = stop - start
            path = outdir / f"{spec['stem']}.{shard_id}.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["quantity"] = quantity
                handle.attrs["shard_id"] = int(shard_id)
                handle.attrs["global_start"] = int(start)
                handle.attrs["global_stop"] = int(stop)
                handle.attrs["modes_label"] = mode_label

                if quantity == "derived":
                    write_string_dataset(handle, "modes", [str(item) for item in modes])
                else:
                    handle.create_dataset("modes", data=np.asarray(modes))
                write_string_dataset(handle, "parameter_names", spec["inputs"])
                handle.create_dataset(
                    "parameters",
                    shape=(len(spec["inputs"]), local_n),
                    dtype=np.float64,
                    chunks=(max(1, len(spec["inputs"])), 1),
                    compression=compression,
                    fillvalue=np.nan,
                )
                handle.create_dataset(
                    "spectra",
                    shape=(len(modes), local_n),
                    dtype=spectra_dtype,
                    chunks=(max(1, len(modes)), 1),
                    compression=compression,
                    fillvalue=np.nan,
                )
                handle.create_dataset("indices", data=-np.ones(local_n, dtype=np.int64))

    write_metadata_file(
        outdir=outdir,
        config_path=config_path,
        config=config,
        n_samples=n_samples_final,
        n_shards=n_shards_final,
        created_quantities=selected_quantities,
        seed=seed,
        dtype_name=dtype_name,
    )


def get_effective_quantities(config: Mapping[str, Any], outdir: Path, requested: Optional[Sequence[str]]) -> List[str]:
    if requested:
        return list(requested)
    metadata_path = outdir / "dataset_metadata.json"
    if metadata_path.exists():
        metadata = read_metadata_file(outdir)
        created = metadata.get("created_quantities")
        if isinstance(created, list) and created:
            return [str(item) for item in created]
    return infer_created_quantities(config)


def build_scalar_context(
    sampled_names: Sequence[str],
    sampled_values: np.ndarray,
    derived_exprs: Mapping[str, str],
    fixed: Mapping[str, float],
) -> Dict[str, float]:
    context = {name: float(sampled_values[i]) for i, name in enumerate(sampled_names)}
    context.update({name: float(value) for name, value in fixed.items()})
    for name, expr in derived_exprs.items():
        context[name] = evaluate_lambda_scalar(expr, context)
    return context


def build_transfer_redshift_nodes(zmax: float, n_nodes: int) -> np.ndarray:
    if zmax <= 0.0:
        return np.array([0.0], dtype=np.float64)
    if n_nodes < 2:
        n_nodes = 2
    x = np.linspace(0.0, 1.0, n_nodes, dtype=np.float64)
    z = np.expm1(np.log1p(zmax) * x)
    z[0] = 0.0
    z[-1] = float(zmax)
    return np.unique(z)


def spherical_tophat_window(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1.0e-4
    xs = x[small]
    out[small] = 1.0 - xs * xs / 10.0 + xs**4 / 280.0
    xl = x[~small]
    out[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl**3)
    return out


def sigma_r_from_pk(kh: np.ndarray, pk: np.ndarray, radius_hmpc: float) -> np.ndarray:
    kh = np.asarray(kh, dtype=np.float64)
    pk = np.asarray(pk, dtype=np.float64)
    if pk.ndim == 1:
        pk = pk[None, :]
    x = kh * float(radius_hmpc)
    window = spherical_tophat_window(x)
    integrand = (kh[None, :] ** 3) * pk * (window[None, :] ** 2) / (2.0 * np.pi**2)
    sigma2 = np.trapz(integrand, x=np.log(kh), axis=1)
    sigma2 = np.clip(sigma2, a_min=0.0, a_max=None)
    return np.sqrt(sigma2)


def interpolate_1d(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    kind = "cubic" if len(x_nodes) >= 4 else "linear"
    interpolator = interp1d(
        x_nodes,
        y_nodes,
        kind=kind,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return np.asarray(interpolator(x_eval), dtype=np.float64)


def configure_camb_and_compute(context: Mapping[str, float], specs: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    import camb
    from camb import model

    extra = dict(config.get("emulated_code", {}).get("extra_args", {}) or {})
    gen = get_generator_config(config)

    quantities = set(specs.keys())
    need_cmb = any(name.startswith("Cl/") for name in quantities)
    need_background = bool({"Hubble", "DA"} & quantities) or ("derived" in quantities)
    need_sigma8_quantity = "sigma8" in quantities
    need_sigma8_scalar = "derived" in quantities and "sigma8_0" in [str(x) for x in specs["derived"]["modes_array"]]
    need_pk_lin = "Pk/lin" in quantities
    need_pk_nonlin = bool({"Pk/nlboost", "Pk/nonlin"} & quantities)
    need_matter = need_sigma8_quantity or need_sigma8_scalar or need_pk_lin or need_pk_nonlin

    dark_energy_model = str(extra.get("dark_energy_model", "ppf"))
    recombination_model = extra.get("recombination_model")
    lens_potential_accuracy = int(extra.get("lens_potential_accuracy", 0))
    ctb_unit = extra.get("CMB_unit", "muK")
    raw_cl = bool(extra.get("raw_cl", False))

    pars = camb.CAMBparams()
    try:
        if recombination_model:
            pars.set_classes(dark_energy_model=dark_energy_model, recombination_model=str(recombination_model))
        else:
            pars.set_classes(dark_energy_model=dark_energy_model)
    except Exception as exc:
        if recombination_model:
            raise RuntimeError(
                f"Your CAMB build does not expose recombination_model={recombination_model!r}. "
                f"Build CAMB with that module or change emulated_code.extra_args.recombination_model explicitly."
            ) from exc
        raise

    pars.set_dark_energy(
        w=float(context.get("w0", -1.0)),
        wa=float(context.get("wa", 0.0)),
        dark_energy_model=dark_energy_model,
    )
    pars.set_cosmology(
        H0=float(context["H0"]),
        ombh2=float(context["ombh2"]),
        omch2=float(context["omch2"]),
        tau=float(context.get("tau", 0.054)),
        mnu=float(context.get("mnu", 0.06)),
        nnu=float(context.get("N_eff", extra.get("nnu", 3.044))),
        num_massive_neutrinos=int(extra.get("num_massive_neutrinos", 1)),
        neutrino_hierarchy=str(extra.get("neutrino_hierarchy", "degenerate")),
    )
    pars.InitPower.set_params(As=float(context["As"]), ns=float(context["ns"]))

    if hasattr(pars, "Accuracy"):
        if "AccuracyBoost" in extra:
            pars.Accuracy.AccuracyBoost = float(extra["AccuracyBoost"])
        if "lAccuracyBoost" in extra:
            pars.Accuracy.lAccuracyBoost = float(extra["lAccuracyBoost"])
        if "lSampleBoost" in extra:
            pars.Accuracy.lSampleBoost = float(extra["lSampleBoost"])
    if "DoLateRadTruncation" in extra and hasattr(pars, "DoLateRadTruncation"):
        pars.DoLateRadTruncation = bool(extra["DoLateRadTruncation"])
    if "min_l_logl_sampling" in extra and hasattr(pars, "min_l_logl_sampling"):
        pars.min_l_logl_sampling = int(extra["min_l_logl_sampling"])

    if need_cmb:
        lmax = 2
        for name, spec in specs.items():
            if name.startswith("Cl/"):
                lmax = max(lmax, int(np.max(spec["modes_array"])))
        pars.set_for_lmax(
            lmax,
            lens_potential_accuracy=lens_potential_accuracy,
            lens_margin=int(extra.get("lens_margin", 150)),
        )
    elif hasattr(pars, "WantCls"):
        pars.WantCls = False

    transfer_z = np.array([0.0], dtype=np.float64)
    if need_matter:
        zmax_needed = 0.0
        if need_sigma8_quantity:
            zmax_needed = max(zmax_needed, float(np.max(specs["sigma8"]["modes_array"])))
        if need_pk_lin or need_pk_nonlin:
            zmax_needed = max(zmax_needed, float(context.get("z_pk", 0.0)))
        transfer_z = build_transfer_redshift_nodes(zmax=zmax_needed, n_nodes=int(gen.get("internal_transfer_nz", 256)))
        pars.set_matter_power(
            redshifts=transfer_z,
            kmax=float(extra.get("kmax", 100.0)),
            k_per_logint=int(extra.get("k_per_logint", 0)),
            nonlinear=need_pk_nonlin,
            silent=True,
        )

    # Final non-linear settings: start from no non-linear matter, then add what is needed.
    pars.NonLinear = model.NonLinear_none
    if need_pk_nonlin:
        pars.NonLinear = model.NonLinear_pk
    if need_cmb and lens_potential_accuracy > 0:
        pars.set_nonlinear_lensing(True)
    if need_pk_nonlin or (need_cmb and lens_potential_accuracy > 0):
        pars.NonLinearModel.set_params(
            halofit_version=str(extra.get("halofit_version", "mead2020_feedback")),
            HMCode_logT_AGN=float(context.get("logT_AGN", 7.8)),
        )

    need_results = need_cmb or need_matter or ("derived" in quantities)
    if need_results:
        results = camb.get_results(pars)
    else:
        results = camb.get_background(pars, no_thermo=False)

    output: Dict[str, np.ndarray] = {}

    if need_cmb:
        lmax = max(int(np.max(spec["modes_array"])) for name, spec in specs.items() if name.startswith("Cl/"))
        cl_dict = results.get_cmb_power_spectra(
            pars,
            lmax=lmax,
            spectra=["lensed_scalar"],
            CMB_unit=ctb_unit,
            raw_cl=raw_cl,
        )
        lensed = np.asarray(cl_dict["lensed_scalar"], dtype=np.float64)
        pp = np.asarray(results.get_lens_potential_cls(lmax=lmax, CMB_unit=ctb_unit, raw_cl=raw_cl), dtype=np.float64)
        if "Cl/tt" in specs:
            output["Cl/tt"] = lensed[2 : lmax + 1, 0]
        if "Cl/ee" in specs:
            output["Cl/ee"] = lensed[2 : lmax + 1, 1]
        if "Cl/bb" in specs:
            output["Cl/bb"] = lensed[2 : lmax + 1, 2]
        if "Cl/te" in specs:
            output["Cl/te"] = lensed[2 : lmax + 1, 3]
        if "Cl/pp" in specs:
            output["Cl/pp"] = pp[2 : lmax + 1, 0]

    pk_lin_interp = None
    pk_nonlin_interp = None
    sigma8_nodes = None
    if need_matter:
        pk_lin_interp = results.get_matter_power_interpolator(
            nonlinear=False,
            hubble_units=True,
            k_hunit=True,
            log_interp=True,
        )
        if need_pk_nonlin:
            pk_nonlin_interp = results.get_matter_power_interpolator(
                nonlinear=True,
                hubble_units=True,
                k_hunit=True,
                log_interp=True,
            )

        if need_sigma8_quantity:
            sigma_cfg = dict(gen.get("sigma8_integration", {}) or {})
            sigma_k = np.logspace(
                np.log10(float(sigma_cfg.get("kmin", 1.0e-4))),
                np.log10(float(sigma_cfg.get("kmax", 50.0))),
                int(sigma_cfg.get("nk", 600)),
                dtype=np.float64,
            )
            sigma8_nodes = np.array(
                [sigma_r_from_pk(sigma_k, np.asarray(pk_lin_interp.P(z, sigma_k), dtype=np.float64), radius_hmpc=8.0)[0] for z in transfer_z],
                dtype=np.float64,
            )
            sigma_z = np.asarray(specs["sigma8"]["modes_array"], dtype=np.float64)
            output["sigma8"] = interpolate_1d(transfer_z, sigma8_nodes, sigma_z)

        if need_pk_lin:
            k_eval = np.asarray(specs["Pk/lin"]["modes_array"], dtype=np.float64)
            output["Pk/lin"] = np.asarray(pk_lin_interp.P(float(context["z_pk"]), k_eval), dtype=np.float64)

        if need_pk_nonlin:
            zpk = float(context["z_pk"])
            if "Pk/nlboost" in specs:
                k_eval = np.asarray(specs["Pk/nlboost"]["modes_array"], dtype=np.float64)
                p_lin = np.asarray(pk_lin_interp.P(zpk, k_eval), dtype=np.float64)
                p_nl = np.asarray(pk_nonlin_interp.P(zpk, k_eval), dtype=np.float64)
                output["Pk/nlboost"] = p_nl / p_lin - 1.0
            if "Pk/nonlin" in specs:
                k_eval = np.asarray(specs["Pk/nonlin"]["modes_array"], dtype=np.float64)
                output["Pk/nonlin"] = np.asarray(pk_nonlin_interp.P(zpk, k_eval), dtype=np.float64)

    if "Hubble" in specs:
        z = np.asarray(specs["Hubble"]["modes_array"], dtype=np.float64)
        output["Hubble"] = np.asarray(results.hubble_parameter(z), dtype=np.float64)

    if "DA" in specs:
        z = np.asarray(specs["DA"]["modes_array"], dtype=np.float64)
        output["DA"] = np.asarray(results.angular_diameter_distance(z), dtype=np.float64)

    if "derived" in specs:
        derived_names = [str(name) for name in specs["derived"]["modes_array"]]
        camb_derived = results.get_derived_params() if hasattr(results, "get_derived_params") else {}
        derived_map: Dict[str, float] = {
            "thetastar": float(camb_derived.get("thetastar", np.nan)),
            "sigma8_0": float(results.get_sigma8_0()) if need_matter else np.nan,
            "zstar": float(camb_derived.get("zstar", np.nan)),
            "rstar": float(camb_derived.get("rstar", np.nan)),
            "zdrag": float(camb_derived.get("zdrag", np.nan)),
            "rdrag": float(camb_derived.get("rdrag", np.nan)),
            "YHe": float(getattr(pars, "YHe", np.nan)),
            "zrei": float(pars.get_zrei()) if hasattr(pars, "get_zrei") else np.nan,
        }
        output["derived"] = np.array([derived_map[name] for name in derived_names], dtype=np.float64)

    missing = sorted(set(specs.keys()) - set(output.keys()))
    if missing:
        raise RuntimeError(f"Did not compute all requested quantities. Missing: {missing}")

    return output


def open_quantity_files(outdir: Path, specs: Mapping[str, Mapping[str, Any]], shard_id: int) -> Dict[str, h5py.File]:
    files: Dict[str, h5py.File] = {}
    for quantity, spec in specs.items():
        path = outdir / f"{spec['stem']}.{shard_id}.hdf5"
        if not path.exists():
            raise FileNotFoundError(f"Missing quantity file {path}. Did you initialize this quantity?")
        files[quantity] = h5py.File(path, "r+")
    return files


def close_quantity_files(files: Mapping[str, h5py.File]) -> None:
    for handle in files.values():
        handle.close()


def run_single_shard(config_path: Path, config: Mapping[str, Any], outdir: Path, shard_id: int, quantities: Optional[Sequence[str]], resume: bool) -> None:
    metadata = read_metadata_file(outdir)
    n_samples = int(metadata["n_samples"])
    n_shards = int(metadata["n_shards"])
    if shard_id < 0 or shard_id >= n_shards:
        raise ValueError(f"shard_id must be in [0, {n_shards - 1}]")

    effective_quantities = get_effective_quantities(config, outdir, quantities)
    specs = select_networks(config, effective_quantities)
    shard_ranges = build_shard_ranges(n_samples, n_shards)
    start, stop = shard_ranges[shard_id]

    params_path = outdir / "parameters.hdf5"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters file: {params_path}")

    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters block missing")
    _, derived_exprs, fixed, _ = split_parameter_blocks(parameters)

    with h5py.File(params_path, "r") as params_handle:
        sampled_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in params_handle["sampled_parameter_names"][...]]
        sampled_table = np.asarray(params_handle["sampled_parameters"][...], dtype=np.float64)

    quantity_files = open_quantity_files(outdir, specs, shard_id)
    fail_log = outdir / f"failed_samples.{shard_id}.log"
    if fail_log.exists() and not resume:
        fail_log.unlink()

    try:
        total_local = stop - start
        for local_idx, global_idx in enumerate(range(start, stop)):
            if resume:
                already_done = True
                for qhandle in quantity_files.values():
                    if int(qhandle["indices"][local_idx]) < 0:
                        already_done = False
                        break
                if already_done:
                    continue

            context = build_scalar_context(
                sampled_names=sampled_names,
                sampled_values=sampled_table[:, global_idx],
                derived_exprs=derived_exprs,
                fixed=fixed,
            )

            try:
                computed = configure_camb_and_compute(context=context, specs=specs, config=config)
            except Exception as exc:
                with fail_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"global_index={global_idx} local_index={local_idx} error={exc}\n")
                    handle.write(traceback.format_exc())
                    handle.write("\n")
                continue

            for quantity, spec in specs.items():
                qhandle = quantity_files[quantity]
                qhandle["parameters"][:, local_idx] = np.asarray([context[name] for name in spec["inputs"]], dtype=np.float64)
                qhandle["spectra"][:, local_idx] = np.asarray(computed[quantity], dtype=qhandle["spectra"].dtype)
                qhandle["indices"][local_idx] = int(global_idx)

            if (local_idx + 1) % 25 == 0 or (local_idx + 1) == total_local:
                for qhandle in quantity_files.values():
                    qhandle.flush()
                print(
                    f"shard {shard_id}: processed {local_idx + 1}/{total_local} samples",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        close_quantity_files(quantity_files)


def report_status(config: Mapping[str, Any], outdir: Path, quantities: Optional[Sequence[str]]) -> None:
    metadata = read_metadata_file(outdir)
    n_samples = int(metadata["n_samples"])
    n_shards = int(metadata["n_shards"])
    effective_quantities = get_effective_quantities(config, outdir, quantities)
    specs = select_networks(config, effective_quantities)

    print(f"dataset: {outdir}")
    print(f"samples: {n_samples}")
    print(f"shards:  {n_shards}")
    for quantity, spec in specs.items():
        filled = 0
        total = 0
        for shard_id in range(n_shards):
            path = outdir / f"{spec['stem']}.{shard_id}.hdf5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as handle:
                indices = np.asarray(handle["indices"][...], dtype=np.int64)
                filled += int(np.count_nonzero(indices >= 0))
                total += int(indices.size)
        print(f"{quantity:12s} {filled:8d} / {total:8d}")


def cmd_init(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    outdir = resolve_outdir(config, args.outdir)
    initialize_dataset(
        config_path=config_path,
        config=config,
        outdir=outdir,
        quantities=parse_quantity_arg(args.quantities),
        n_samples=args.n_samples,
        n_shards=args.n_shards,
        overwrite=bool(args.overwrite),
    )


def cmd_run_shard(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    outdir = resolve_outdir(config, args.outdir)
    run_single_shard(
        config_path=config_path,
        config=config,
        outdir=outdir,
        shard_id=int(args.shard),
        quantities=parse_quantity_arg(args.quantities),
        resume=bool(args.resume),
    )


def cmd_run_all(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    outdir = resolve_outdir(config, args.outdir)
    metadata_path = outdir / "dataset_metadata.json"
    requested_quantities = parse_quantity_arg(args.quantities)

    if not metadata_path.exists():
        initialize_dataset(
            config_path=config_path,
            config=config,
            outdir=outdir,
            quantities=requested_quantities,
            n_samples=args.n_samples,
            n_shards=args.n_shards,
            overwrite=bool(args.overwrite),
        )

    metadata = read_metadata_file(outdir)
    n_shards = int(metadata["n_shards"])
    for shard_id in range(n_shards):
        run_single_shard(
            config_path=config_path,
            config=config,
            outdir=outdir,
            shard_id=shard_id,
            quantities=requested_quantities,
            resume=bool(args.resume),
        )


def cmd_status(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    outdir = resolve_outdir(config, args.outdir)
    report_status(config=config, outdir=outdir, quantities=parse_quantity_arg(args.quantities))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the parameter table and pre-allocate HDF5 files")
    p_init.add_argument("config", help="YAML configuration file")
    p_init.add_argument("--outdir", default=None, help="output directory; defaults to the config path field")
    p_init.add_argument("--quantities", default=None, help="comma-separated subset of quantities to initialize")
    p_init.add_argument("--n-samples", type=int, default=None, help="override samples.Ntraining")
    p_init.add_argument("--n-shards", type=int, default=None, help="override generator.n_shards")
    p_init.add_argument("--overwrite", action="store_true", help="remove an existing output directory before init")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run-shard", help="compute one shard of the training set")
    p_run.add_argument("config", help="YAML configuration file")
    p_run.add_argument("--outdir", default=None, help="output directory; defaults to the config path field")
    p_run.add_argument("--shard", type=int, required=True, help="shard id to generate")
    p_run.add_argument("--quantities", default=None, help="comma-separated subset of quantities to fill")
    p_run.add_argument("--resume", action="store_true", help="skip samples already present in all selected quantities")
    p_run.set_defaults(func=cmd_run_shard)

    p_all = sub.add_parser("run-all", help="initialize if needed and compute every shard sequentially")
    p_all.add_argument("config", help="YAML configuration file")
    p_all.add_argument("--outdir", default=None, help="output directory; defaults to the config path field")
    p_all.add_argument("--quantities", default=None, help="comma-separated subset of quantities to fill")
    p_all.add_argument("--n-samples", type=int, default=None, help="override samples.Ntraining when initializing")
    p_all.add_argument("--n-shards", type=int, default=None, help="override generator.n_shards when initializing")
    p_all.add_argument("--overwrite", action="store_true", help="remove an existing output directory before init")
    p_all.add_argument("--resume", action="store_true", help="skip samples already present in all selected quantities")
    p_all.set_defaults(func=cmd_run_all)

    p_status = sub.add_parser("status", help="show how many samples are filled")
    p_status.add_argument("config", help="YAML configuration file")
    p_status.add_argument("--outdir", default=None, help="output directory; defaults to the config path field")
    p_status.add_argument("--quantities", default=None, help="comma-separated subset of quantities to report")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
