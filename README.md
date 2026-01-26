# JuliaLBM - GPU-Accelerated Lattice Boltzmann CFD Solver

A high-performance, GPU-accelerated Computational Fluid Dynamics (CFD) solver based on the Lattice Boltzmann Method (LBM). Features D3Q27 velocity discretization, WALE turbulence modeling, multi-level grid refinement, and Bouzidi interpolated boundary conditions for accurate aerodynamic simulations.

![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green.svg)

---

## Table of Contents

- [Features](#features)
- [Technical Characteristics](#technical-characteristics)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Setting Up Cases](#setting-up-cases)
- [Configuration Reference](#configuration-reference)
- [Post-Processing](#post-processing)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Features

- **D3Q27 Lattice**: Full 27-velocity model for improved isotropy and accuracy
- **WALE Turbulence Model**: Wall-Adapting Local Eddy-viscosity for LES
- **Wall-Modeled LES (WMLES)**: Enables high Reynolds number simulations
- **Multi-Level Grid Refinement**: Automatic nested grid generation around geometry
- **Bouzidi Boundary Conditions**: Second-order accurate interpolated bounce-back
- **GPU Acceleration**: CUDA-based computation via KernelAbstractions.jl
- **Force & Moment Computation**: Aerodynamic coefficients (Cd, Cl, Cm) via momentum exchange
- **VTK Output**: ParaView-compatible unstructured mesh export
- **Batch Case Execution**: Run multiple cases sequentially

---

## Technical Characteristics

### Comparison with State-of-the-Art LBM Solvers

| Feature | JuliaLBM | Traditional LBM | Commercial CFD |
|---------|----------|-----------------|----------------|
| **Velocity Set** | D3Q27 (27 velocities) | D3Q19 typical | N/A (FVM/FEM) |
| **Collision Operator** | Regularized BGK with WALE | BGK or MRT | Various RANS/LES |
| **Boundary Treatment** | Bouzidi interpolated (2nd order) | Simple bounce-back (1st order) | Body-fitted mesh |
| **Grid Refinement** | Block-structured multi-level | Uniform or octree | Unstructured |
| **Wall Modeling** | Equilibrium stress (log-law) | Often none | Wall functions |
| **Memory Pattern** | A-B (dual buffer) | Various | N/A |

### Key Technical Innovations

1. **Sparse Block Storage**: Only active blocks near geometry are allocated, reducing memory by 60-80% compared to full-domain approaches.

2. **Bouzidi Sparse Implementation**: Q-values stored only for boundary cells with coordinate lists, minimizing memory overhead while maintaining second-order accuracy.

3. **WALE Turbulence Model**: Unlike Smagorinsky, WALE correctly predicts zero eddy viscosity in pure shear and near walls without ad-hoc damping functions:
   ```
   ν_t = (C_w Δ)² × (S^d_ij S^d_ij)^(3/2) / [(S_ij S_ij)^(5/2) + (S^d_ij S^d_ij)^(5/4)]
   ```

4. **Regularized Collision**: Non-equilibrium stress tensor reconstruction improves stability at high Reynolds numbers while maintaining accuracy.

5. **Multi-Level Time Stepping**: Recursive 2:1 time step ratio between refinement levels ensures proper physics coupling.

### Lattice Boltzmann Fundamentals

The solver implements the lattice Boltzmann equation:
```
f_i(x + c_i Δt, t + Δt) = f_i(x, t) - (f_i - f_i^eq)/τ + F_i
```

Where:
- `f_i`: Distribution function for velocity direction i
- `f_i^eq`: Maxwell-Boltzmann equilibrium
- `τ`: Relaxation time (related to viscosity: ν = c_s² (τ - 0.5) Δt)
- `F_i`: External forcing term

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- **VRAM**: Minimum 8 GB recommended (scales with mesh size)
- **RAM**: 16 GB+ recommended

### Software
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, macOS
- **Julia**: Version 1.9 or higher
- **CUDA Toolkit**: Version 11.0+ (for GPU acceleration)

---

## Installation

### Step 1: Install Julia

#### Linux (Ubuntu/Debian)
```bash
# Download Julia
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.2-linux-x86_64.tar.gz

# Extract
tar -xzf julia-1.10.2-linux-x86_64.tar.gz

# Add to PATH (add to ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/julia-1.10.2/bin"
```

#### Windows
1. Download the installer from [julialang.org/downloads](https://julialang.org/downloads/)
2. Run the installer and check "Add Julia to PATH"
3. Restart your terminal/PowerShell

#### macOS
```bash
# Using Homebrew
brew install julia

# Or download from julialang.org
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/yourusername/JuliaLBM.git
cd JuliaLBM
```

### Step 3: Install Dependencies

Run the dependency installer script:

```bash
julia src/00_First_time_install_packages.jl
```

This installs:
- `KernelAbstractions` - Backend-agnostic GPU kernels
- `CUDA` - NVIDIA GPU support
- `Adapt` - CPU/GPU data transfer
- `StaticArrays` - High-performance small arrays
- `YAML` - Configuration file parsing
- `WriteVTK` - ParaView output

**Note**: First-time compilation may take 5-15 minutes.

### Step 4: Verify Installation

```bash
julia -e "using CUDA; CUDA.functional() && println(\"GPU: \", CUDA.name(CUDA.device()))"
```

Expected output: `GPU: NVIDIA GeForce RTX XXXX` (or similar)

---

## Quick Start

### Running Your First Simulation

1. **Prepare a case** (example: Stanford Bunny is included):
   ```
   CASES/
   └── Stanford_bunny/
       ├── config.yaml    # Configuration file
       └── bunny.stl      # Geometry file
   ```

2. **Edit `cases_to_run.yaml`** to specify which cases to run:
   ```yaml
   case_folders:
     - "Stanford_bunny"
   ```

3. **Run the solver**:
   ```bash
   julia src/_RUN_CASE_.jl
   ```

4. **Monitor progress**: The solver prints timestep, physical time, lattice velocity, density, MLUPS (Million Lattice Updates Per Second), and aerodynamic coefficients.

5. **Results** are saved in `CASES/<case_name>/RESULTS/`:
   - `flow_XXXXXX.vtu` - VTK files for visualization
   - `convergence.csv` - Time history of solver metrics
   - `forces.csv` - Aerodynamic force/moment history

---

## Setting Up Cases

### Directory Structure

```
CASES/
└── My_New_Case/
    ├── config.yaml      # Required: solver configuration
    └── geometry.stl     # Required: STL geometry file
```

### Creating a New Case

1. **Create a folder** in `CASES/` with your case name
2. **Add your STL file** (ensure it's watertight and in meters or set `stl_scale`)
3. **Create `config.yaml`** (copy from an existing case and modify)
4. **Add case to `cases_to_run.yaml`**

### STL File Guidelines

- **Format**: Binary or ASCII STL
- **Units**: Preferably meters (use `stl_scale` to convert)
- **Quality**: Watertight mesh, no self-intersections
- **Orientation**: Flow typically along +X axis
- **Origin**: Position doesn't matter (auto-centered in domain)

---

## Configuration Reference

The `config.yaml` file controls all simulation parameters. Below is a detailed reference:

### Basic Parameters

```yaml
basic:
  # GEOMETRY
  stl_file: "model.stl"      # STL filename (in case folder)
  stl_scale: 1.0             # Scale factor (0.001 for mm→m)
  
  # MESH RESOLUTION
  surface_resolution: 200    # Cells per reference length (higher = finer mesh)
                             # Memory scales as N³, compute as N⁴
                             # Typical values: 100-300 (coarse), 500-1000 (fine)
  
  num_levels: 5              # Grid refinement levels
                             # 0 = auto-compute based on domain
                             # Each level doubles resolution near geometry
  
  # REFERENCE VALUES (for coefficient calculation only)
  reference_area_of_full_model: 1.0   # [m²] Planform area for Cd, Cl
  reference_chord: 0.5                 # [m] Chord for Cm calculation
  reference_length_for_meshing: 1.0   # [m] Length for Reynolds number
  reference_dimension: "x"             # Which STL dimension is reference length
```

**Understanding `surface_resolution`**:
- This is the most critical parameter for accuracy vs. cost
- Represents cells spanning the reference length at the finest level
- For WMLES, target y+ of 30-100 at the first cell
- Rule of thumb: Start with 200, increase until results converge

### Physics Parameters

```yaml
  fluid:
    density: 1.225                    # [kg/m³] Air at sea level
    kinematic_viscosity: 1.5e-5       # [m²/s] Air ≈ 1.5e-5, Water ≈ 1.0e-6
  
  flow:
    velocity: 10.0                    # [m/s] Freestream velocity
```

**Reynolds Number**: Computed automatically as `Re = velocity × reference_length / kinematic_viscosity`

### Simulation Control

```yaml
  simulation:
    steps: 10000           # Total timesteps
    ramp_steps: 2000       # Velocity ramp-up period (prevents instability)
    output_freq: 2500      # VTK output every N steps
    output_dir: "RESULTS"  # Output folder name
    
    output_fields:         # Which fields to save
      density: false
      velocity: true
      velocity_magnitude: true
      vorticity: false
      obstacle: true
      level: true
      bouzidi: false
```

**Timestep Estimation**: Physical time per step ≈ `dx_fine / (velocity × 100)`. For 10,000 steps with fine mesh, expect ~0.1-1.0 seconds of physical time.

### Advanced Numerics

```yaml
advanced:
  numerics:
    u_lattice: 0.01        # Lattice velocity (Ma_lattice ≈ u_lattice/0.577)
                           # Lower = more stable, more accurate
                           # Range: 0.01 (precision) to 0.08 (fast)
                           # Compressibility error ∝ u_lattice²
    
    c_wale: 0.50           # WALE model constant
                           # 0.325 = theoretical, 0.5-0.6 = stable for LBM
    
    tau_min: 0.500001      # Minimum relaxation time (τ > 0.5 required)
                           # Prevents zero/negative viscosity
    
    tau_safety_factor: 1.0 # Multiplier for τ calculation
                           # >1.0 = more conservative (coarser effective mesh)
```

**Understanding τ (Tau)**:
- Related to viscosity: `ν = (τ - 0.5) / 3`
- τ → 0.5 means ν → 0 (inviscid, unstable)
- τ > 0.55 recommended for stability
- High Re flows naturally push τ toward 0.5

### Wall Modeling

```yaml
  high_re:
    wall_model:
      enabled: true              # Activate WMLES wall model
      type: "equilibrium"        # Log-law based wall stress
      y_plus_target: 100.0       # Target y+ for first cell
```

**When to Enable**:
- Re > 10⁵ with limited resolution
- When direct wall resolution is impractical
- Reduces mesh requirements by 10-100×

### Domain Sizing

```yaml
  domain:
    upstream: 0.75       # Distance upstream (× reference_length)
    downstream: 1.5      # Distance downstream (wake capture)
    lateral: 0.75        # Side clearance
    height: 0.75         # Top/bottom clearance
    sponge_thickness: 0.10   # Boundary absorption layer (fraction)
```

**Recommendations**:
- Upstream: 0.5-1.0 L (inlet turbulence dissipation)
- Downstream: 1.5-3.0 L (wake development)
- Sponge: 0.05-0.15 (absorbs reflections)

### Boundary Conditions

```yaml
  boundary:
    method: "bouzidi"       # "bouzidi" (2nd order) or "bounce_back" (1st order)
    bouzidi_levels: 1       # How many fine levels use Bouzidi
    use_float16_qmap: true  # Memory optimization for Q-values
    q_min_threshold: 0.001  # Minimum Q for boundary detection
  
  refinement:
    symmetric_analysis: false   # true = symmetry plane at Y=0
    block_size: 8               # Cells per block (8 optimal for GPU)
    margin: 2                   # Buffer blocks around geometry
```

### Force Computation

```yaml
  forces:
    enabled: true              # Compute aerodynamic forces
    output_freq: 0             # 0 = match diagnostics frequency
    moment_center: [0.25, 0.0, 0.0]  # Moment reference point
                                      # [x/chord, y/span, z/thickness]
                                      # 0.25 = quarter-chord (typical)
```

### GPU Optimization

```yaml
  gpu:
    async_depth: 8        # Timesteps queued before sync
                          # Higher = faster, less responsive
    use_streams: true     # CUDA stream parallelism
    prefetch_neighbors: true  # Memory prefetching
```

---

## Post-Processing

### Viewing Results in ParaView

1. **Open ParaView** (download from [paraview.org](https://www.paraview.org/))

2. **Load VTK files**: File → Open → Select `flow_*.vtu` files

3. **Recommended visualizations**:
   - **Velocity magnitude**: Color by "VelocityMagnitude"
   - **Streamlines**: Filters → Stream Tracer
   - **Iso-surfaces**: Filters → Contour (Q-criterion for vortices)
   - **Slices**: Filters → Slice

4. **Animation**: Use the time controls to animate through timesteps

### Analyzing Force Data

The `forces.csv` file contains:

| Column | Description |
|--------|-------------|
| `Step` | Timestep number |
| `Time_phys` | Physical time [s] |
| `Fx_N, Fy_N, Fz_N` | Forces [N] |
| `Mx_Nm, My_Nm, Mz_Nm` | Moments [N·m] |
| `Cd, Cl, Cs` | Drag, Lift, Side force coefficients |
| `Cmx, Cmy, Cmz` | Moment coefficients |
| `U_inlet` | Inlet velocity (for ramp tracking) |

**Python Analysis Example**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('RESULTS/forces.csv')

# Plot coefficients vs time
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(df['Time_phys'], df['Cd'], label='Cd')
axes[0].plot(df['Time_phys'], df['Cl'], label='Cl')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Coefficient')
axes[0].legend()
axes[0].set_title('Aerodynamic Coefficients')

axes[1].plot(df['Time_phys'], df['Cmy'], label='Cm (pitch)')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Moment Coefficient')
axes[1].legend()

plt.tight_layout()
plt.savefig('coefficients.png', dpi=150)
```

### Convergence Monitoring

Check `convergence.csv` for:
- `Rho_min/Rho_max`: Should stay near 1.0 (±5%)
- `U_max_lat`: Should stay below 0.3 (compressibility limit)
- `MLUPS`: Performance metric (higher = faster)

**Warning Signs**:
- Rho dropping below 0.5 → simulation unstable
- Rho oscillating wildly → reduce `u_lattice` or increase `ramp_steps`
- MLUPS dropping → memory issues or thermal throttling

---

## Performance

### Typical Performance (NVIDIA RTX 4090)

| Case Size | Cells | VRAM | MLUPS | Real-time Factor |
|-----------|-------|------|-------|------------------|
| Small (100³) | 1M | 0.5 GB | 800+ | ~100× |
| Medium (200³) | 8M | 2 GB | 600 | ~50× |
| Large (400³) | 64M | 12 GB | 400 | ~20× |
| Very Large (600³) | 216M | 24 GB | 300 | ~10× |

**MLUPS** = Million Lattice Updates Per Second

### Memory Estimation

Approximate VRAM usage:
```
VRAM (GB) ≈ Total_Cells × 300 bytes / 10⁹
```

Where `Total_Cells = N_blocks × 8³` summed across all levels.

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `surface_resolution`
- Reduce `num_levels`
- Enable `use_float16_qmap: true`

**Simulation explodes (NaN/Inf)**
- Increase `ramp_steps` (try 5000+)
- Reduce `u_lattice` (try 0.005)
- Check STL quality (watertight?)
- Ensure `tau_min` > 0.5

**τ warnings**
- Expected at high Re
- Enable `wall_model` if τ < 0.51 frequently
- Reduce resolution if τ approaches 0.5000001

**Slow first run**
- Julia compiles on first execution
- Subsequent runs are much faster
- Use `julia --project=.` to maintain precompilation

**No GPU detected**
- Verify CUDA installation: `nvidia-smi`
- Check Julia CUDA: `julia -e "using CUDA; CUDA.versioninfo()"`
- Ensure GPU drivers are up to date

---

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{julialb,
  title = {JuliaLBM: GPU-Accelerated Lattice Boltzmann CFD Solver},
  year = {2024},
  url = {https://github.com/yourusername/JuliaLBM}
}
```

---

## License

This project is licensed under the 

---

