# nuw0waCDM CAMB training-data generator for CosmoPower

Files included here:

- `nuw0wa_camb_package.yaml`: paper-style package/config file.
- `generate_nuw0wa_training_data.py`: main dataset builder.
- `1_create_params_nuw0wa_camb.py`: thin wrapper around `init`.
- `2_create_spectra_nuw0wa_camb.py`: thin wrapper around `run-shard`.
- `3_check_dataset_nuw0wa_camb.py`: thin wrapper around `status`.
- `run_nuw0wa_generation.sh`: sequential convenience launcher.

## What this gives you

This is designed to match the paper's Section 3 layout:

- one `parameters.hdf5` file containing the Latin-hypercube samples;
- one HDF5 file per quantity per shard, e.g. `Cl_tt.0.hdf5`, `Pk_lin.3.hdf5`, `DA.7.hdf5`;
- each quantity file stores the mode vector, emulator input names, emulator inputs, spectra, and the back-reference indices into the parameter file.

## Quantities in the default YAML

The YAML includes the paper-style network entries for:

- `derived`
- `Cl/tt`, `Cl/te`, `Cl/ee`, `Cl/bb`, `Cl/pp`
- `Pk/lin`, `Pk/nlboost`
- `Hubble`, `DA`, `sigma8`

If you do not want all of these, initialize and run only a subset, e.g.

```bash
python generate_nuw0wa_training_data.py init nuw0wa_camb_package.yaml \
  --outdir ./nuw0wa_camb_data \
  --quantities Cl/tt,Cl/te,Cl/ee,Cl/pp,Pk/lin,Pk/nlboost
```

and then generate only that same subset.

## Installation

At minimum you need:

```bash
pip install numpy scipy h5py pyyaml camb
```

## Important CAMB note

The YAML is set to `recombination_model: CosmoRec` to follow the paper's appendix.
Standard CAMB wheels do not necessarily include CosmoRec support. If your CAMB build
raises an error when the script tries to set `CosmoRec`, either:

1. build CAMB with CosmoRec enabled, or
2. explicitly change the YAML field to `Recfast` and accept that this is no longer a strict paper match.

## Recommended workflow

### 1. Create the parameter table and pre-allocate the HDF5 files

```bash
python generate_nuw0wa_training_data.py init nuw0wa_camb_package.yaml --outdir ./nuw0wa_camb_data
```

Equivalent wrapper matching the old `spectra_generation_scripts` style:

```bash
python 1_create_params_nuw0wa_camb.py nuw0wa_camb_package.yaml --outdir ./nuw0wa_camb_data
```

### 2. Generate one shard

```bash
python generate_nuw0wa_training_data.py run-shard nuw0wa_camb_package.yaml \
  --outdir ./nuw0wa_camb_data \
  --shard 0 --resume
```

Equivalent wrapper:

```bash
python 2_create_spectra_nuw0wa_camb.py nuw0wa_camb_package.yaml \
  --outdir ./nuw0wa_camb_data \
  --shard 0 --resume
```

### 3. Run all shards sequentially

```bash
bash run_nuw0wa_generation.sh nuw0wa_camb_package.yaml ./nuw0wa_camb_data
```

### 4. Check status

```bash
python generate_nuw0wa_training_data.py status nuw0wa_camb_package.yaml --outdir ./nuw0wa_camb_data
```

Equivalent wrapper:

```bash
python 3_check_dataset_nuw0wa_camb.py nuw0wa_camb_package.yaml --outdir ./nuw0wa_camb_data
```

## Cluster use

The clean pattern is one job per shard. For ten shards:

```bash
python generate_nuw0wa_training_data.py init nuw0wa_camb_package.yaml --outdir ./nuw0wa_camb_data
```

then submit ten jobs with shard ids `0` through `9`, each calling

```bash
python generate_nuw0wa_training_data.py run-shard nuw0wa_camb_package.yaml \
  --outdir ./nuw0wa_camb_data --shard SHARD_ID --resume
```

## Conventions used here

- CMB spectra are stored using CAMB's default `raw_cl=False` convention, i.e. `D_ell = ell(ell+1) C_ell / 2pi`.
- `Cl/pp` stores CAMB's lensing-potential output convention.
- `Pk/lin` is stored in `(Mpc/h)^3` as a function of `k/h`.
- `Pk/nlboost` is stored as `P_nl / P_lin - 1`.
- `sigma8(z)` is built from the linear matter spectrum.
- `derived` includes scalar values convenient for later bookkeeping; `sigma8_0` is used there to avoid name collision with the `sigma8(z)` quantity.

## Practical notes

- `w0` and `wa` are sampled over a range that can cross `w = -1`; the YAML therefore sets CAMB's dark-energy model to `ppf`.
- The default config includes `logT_AGN` because the paper includes that extra HMCode-2020 parameter for non-linear matter emulators.
- If you are not training `Pk/nlboost`, you can remove `logT_AGN` from the sampled parameter list or just omit that quantity.
