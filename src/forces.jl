"""
FORCES.JL - Aerodynamic Force and Moment Computation

Computes drag, lift, side forces and pitching moment on immersed bodies 
using the momentum exchange method at Bouzidi boundary cells.

Forces:  F = Σ (f_k + f_opp) × c_k  (momentum exchange)
Moments: M = Σ (r × F)              (about moment center)

Output coefficients:
  Cd = Fx / (0.5 × ρ × U² × A)    Drag coefficient
  Cl = Fz / (0.5 × ρ × U² × A)    Lift coefficient
  Cs = Fy / (0.5 × ρ × U² × A)    Side force coefficient
  Cm = My / (0.5 × ρ × U² × A × c) Pitching moment coefficient
"""

using KernelAbstractions
using CUDA
using Adapt
using Printf

# ============================================================================
# FORCE/MOMENT COMPUTATION KERNEL (Sparse - only boundary cells)
# ============================================================================

"""
Compute force and moment contributions at each boundary cell.
Each thread handles one boundary cell.
"""
@kernel function compute_boundary_forces_moments_kernel!(
    Fx_partial, Fy_partial, Fz_partial,  # Output: partial forces
    Mx_partial, My_partial, Mz_partial,  # Output: partial moments
    f_curr,                               # Current distributions
    q_map,                                # Q-values
    cell_block, cell_x, cell_y, cell_z,  # Sparse boundary cell coordinates
    map_x, map_y, map_z,                 # Block coordinate maps
    cx_arr, cy_arr, cz_arr,
    opp_arr,
    q_min_threshold::Float32,
    block_size::Int32,
    dx::Float32,                          # Grid spacing
    moment_cx::Float32,                   # Moment center x
    moment_cy::Float32,                   # Moment center y
    moment_cz::Float32                    # Moment center z
)
    cell_idx = @index(Global)
    
    @inbounds begin
        b_idx = cell_block[cell_idx]
        lx = cell_x[cell_idx]
        ly = cell_y[cell_idx]
        lz = cell_z[cell_idx]
        
        # Block coordinates
        bx = map_x[b_idx]
        by = map_y[b_idx]
        bz = map_z[b_idx]
        
        # Global cell position (physical coordinates)
        px = ((bx - 1) * block_size + lx - 0.5f0) * dx
        py = ((by - 1) * block_size + ly - 0.5f0) * dx
        pz = ((bz - 1) * block_size + lz - 0.5f0) * dx
        
        # Position vector from moment center
        rx = px - moment_cx
        ry = py - moment_cy
        rz = pz - moment_cz
        
        fx = 0.0f0
        fy = 0.0f0
        fz = 0.0f0
        
        # Sum momentum exchange for all directions with walls
        for k in 1:27
            q = Float32(q_map[lx, ly, lz, b_idx, k])
            
            if q > q_min_threshold && q <= 1.0f0
                opp_k = opp_arr[k]
                
                # Momentum exchange: F = (f_k + f_opp) × c_k
                f_k = f_curr[lx, ly, lz, b_idx, k]
                f_opp = f_curr[lx, ly, lz, b_idx, opp_k]
                
                cx = Float32(cx_arr[k])
                cy = Float32(cy_arr[k])
                cz = Float32(cz_arr[k])
                
                momentum = f_k + f_opp
                
                fx += momentum * cx
                fy += momentum * cy
                fz += momentum * cz
            end
        end
        
        Fx_partial[cell_idx] = fx
        Fy_partial[cell_idx] = fy
        Fz_partial[cell_idx] = fz
        
        # Moment: M = r × F
        Mx_partial[cell_idx] = ry * fz - rz * fy
        My_partial[cell_idx] = rz * fx - rx * fz
        Mz_partial[cell_idx] = rx * fy - ry * fx
    end
end

# ============================================================================
# FORCE/MOMENT DATA STRUCTURE
# ============================================================================

"""
Stores force/moment computation results and reference values.
"""
mutable struct ForceData
    # Forces (lattice units)
    Fx_lattice::Float64
    Fy_lattice::Float64
    Fz_lattice::Float64
    
    # Moments (lattice units)
    Mx_lattice::Float64
    My_lattice::Float64
    Mz_lattice::Float64
    
    # Forces (physical units, Newtons)
    Fx_physical::Float64
    Fy_physical::Float64
    Fz_physical::Float64
    
    # Moments (physical units, Newton-meters)
    Mx_physical::Float64
    My_physical::Float64
    Mz_physical::Float64
    
    # Force coefficients
    Cd::Float64  # Drag coefficient (X)
    Cl::Float64  # Lift coefficient (Z)
    Cs::Float64  # Side force coefficient (Y)
    
    # Moment coefficients
    Cmx::Float64  # Roll moment coefficient
    Cmy::Float64  # Pitch moment coefficient
    Cmz::Float64  # Yaw moment coefficient
    
    # Reference values (physical)
    rho_ref::Float64        # Reference density [kg/m³]
    u_ref::Float64          # Reference velocity [m/s]
    area_ref::Float64       # Reference area [m²]
    chord_ref::Float64      # Reference chord [m]
    moment_center::Tuple{Float64, Float64, Float64}  # Moment center [m]
    
    # Unit conversion
    force_scale::Float64    # lattice → physical force
    length_scale::Float64   # lattice → physical length
    
    # Symmetry flag
    symmetric::Bool
end

function ForceData(; rho_ref=1.225, u_ref=10.0, area_ref=1.0, chord_ref=1.0,
                    moment_center=(0.0, 0.0, 0.0), force_scale=1.0, length_scale=1.0,
                    symmetric=false)
    return ForceData(
        0.0, 0.0, 0.0,  # Lattice forces
        0.0, 0.0, 0.0,  # Lattice moments
        0.0, 0.0, 0.0,  # Physical forces
        0.0, 0.0, 0.0,  # Physical moments
        0.0, 0.0, 0.0,  # Force coefficients
        0.0, 0.0, 0.0,  # Moment coefficients
        rho_ref, u_ref, area_ref, chord_ref, moment_center,
        force_scale, length_scale,
        symmetric
    )
end

# ============================================================================
# FORCE/MOMENT COMPUTATION FUNCTION
# ============================================================================

"""
    compute_forces!(force_data, level, f_curr, cx_gpu, cy_gpu, cz_gpu, opp_gpu, ...)

Compute aerodynamic forces and moments on the immersed body.
Only operates on the finest level with Bouzidi boundaries.
"""
function compute_forces!(force_data::ForceData,
                         level,  # BlockLevel with Bouzidi data
                         f_curr,  # Current distribution
                         cx_gpu, cy_gpu, cz_gpu, opp_gpu,
                         q_min_threshold::Float32,
                         backend)
    
    if !level.bouzidi_enabled || level.n_boundary_cells == 0
        # Zero out all values
        force_data.Fx_lattice = 0.0
        force_data.Fy_lattice = 0.0
        force_data.Fz_lattice = 0.0
        force_data.Mx_lattice = 0.0
        force_data.My_lattice = 0.0
        force_data.Mz_lattice = 0.0
        force_data.Fx_physical = 0.0
        force_data.Fy_physical = 0.0
        force_data.Fz_physical = 0.0
        force_data.Mx_physical = 0.0
        force_data.My_physical = 0.0
        force_data.Mz_physical = 0.0
        force_data.Cd = 0.0
        force_data.Cl = 0.0
        force_data.Cs = 0.0
        force_data.Cmx = 0.0
        force_data.Cmy = 0.0
        force_data.Cmz = 0.0
        return (0.0, 0.0, 0.0)
    end
    
    n_boundary = level.n_boundary_cells
    
    # Allocate partial arrays
    Fx_partial = KernelAbstractions.zeros(backend, Float32, n_boundary)
    Fy_partial = KernelAbstractions.zeros(backend, Float32, n_boundary)
    Fz_partial = KernelAbstractions.zeros(backend, Float32, n_boundary)
    Mx_partial = KernelAbstractions.zeros(backend, Float32, n_boundary)
    My_partial = KernelAbstractions.zeros(backend, Float32, n_boundary)
    Mz_partial = KernelAbstractions.zeros(backend, Float32, n_boundary)
    
    # Launch kernel
    mc = force_data.moment_center
    kernel! = compute_boundary_forces_moments_kernel!(backend)
    kernel!(Fx_partial, Fy_partial, Fz_partial,
            Mx_partial, My_partial, Mz_partial,
            f_curr,
            level.bouzidi_q_map,
            level.bouzidi_cell_block,
            level.bouzidi_cell_x,
            level.bouzidi_cell_y,
            level.bouzidi_cell_z,
            level.map_x, level.map_y, level.map_z,
            cx_gpu, cy_gpu, cz_gpu, opp_gpu,
            q_min_threshold,
            Int32(BLOCK_SIZE),
            Float32(level.dx),
            Float32(mc[1]), Float32(mc[2]), Float32(mc[3]),
            ndrange=(n_boundary,))
    
    KernelAbstractions.synchronize(backend)
    
    # Sum partial values
    Fx_lat = Float64(sum(Fx_partial))
    Fy_lat = Float64(sum(Fy_partial))
    Fz_lat = Float64(sum(Fz_partial))
    Mx_lat = Float64(sum(Mx_partial))
    My_lat = Float64(sum(My_partial))
    Mz_lat = Float64(sum(Mz_partial))
    
    # Apply symmetry correction
    if force_data.symmetric
        Fy_lat *= 2.0   # Double Y-force
        Mx_lat *= 2.0   # Double roll moment
        Mz_lat *= 2.0   # Double yaw moment
        # My (pitch) is already full due to symmetry
    end
    
    # Store lattice values
    force_data.Fx_lattice = Fx_lat
    force_data.Fy_lattice = Fy_lat
    force_data.Fz_lattice = Fz_lat
    force_data.Mx_lattice = Mx_lat
    force_data.My_lattice = My_lat
    force_data.Mz_lattice = Mz_lat
    
    # Convert to physical units
    fs = force_data.force_scale
    ls = force_data.length_scale
    
    force_data.Fx_physical = Fx_lat * fs
    force_data.Fy_physical = Fy_lat * fs
    force_data.Fz_physical = Fz_lat * fs
    force_data.Mx_physical = Mx_lat * fs * ls
    force_data.My_physical = My_lat * fs * ls
    force_data.Mz_physical = Mz_lat * fs * ls
    
    # Compute coefficients
    # q = 0.5 × ρ × U²
    q_dyn = 0.5 * force_data.rho_ref * force_data.u_ref^2
    
    # Force reference: q × A
    F_ref = q_dyn * force_data.area_ref
    
    # Moment reference: q × A × c
    M_ref = F_ref * force_data.chord_ref
    
    if F_ref > 1e-10
        force_data.Cd = force_data.Fx_physical / F_ref
        force_data.Cl = force_data.Fz_physical / F_ref
        force_data.Cs = force_data.Fy_physical / F_ref
    else
        force_data.Cd = 0.0
        force_data.Cl = 0.0
        force_data.Cs = 0.0
    end
    
    if M_ref > 1e-10
        force_data.Cmx = force_data.Mx_physical / M_ref
        force_data.Cmy = force_data.My_physical / M_ref
        force_data.Cmz = force_data.Mz_physical / M_ref
    else
        force_data.Cmx = 0.0
        force_data.Cmy = 0.0
        force_data.Cmz = 0.0
    end
    
    return (force_data.Fx_physical, force_data.Fy_physical, force_data.Fz_physical)
end

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

"""
Print force/moment summary to console.
"""
function print_force_summary(force_data::ForceData, step::Int)
    @printf("  Forces @ %d: Cd=%.4f, Cl=%.4f, Cm=%.4f (Fx=%.2e N, Fz=%.2e N, My=%.2e Nm)\n",
            step, force_data.Cd, force_data.Cl, force_data.Cmy,
            force_data.Fx_physical, force_data.Fz_physical, force_data.My_physical)
end

"""
Write force/moment CSV header.
"""
function write_force_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io, "Step,Time_phys,Fx_N,Fy_N,Fz_N,Mx_Nm,My_Nm,Mz_Nm,Cd,Cl,Cs,Cmx,Cmy,Cmz,U_inlet")
    end
end

"""
Append force/moment data to CSV.
"""
function append_force_csv(filepath::String, step::Int, time_phys::Float64, 
                          force_data::ForceData, u_inlet::Float32)
    open(filepath, "a") do io
        @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                step, time_phys,
                force_data.Fx_physical, force_data.Fy_physical, force_data.Fz_physical,
                force_data.Mx_physical, force_data.My_physical, force_data.Mz_physical,
                force_data.Cd, force_data.Cl, force_data.Cs,
                force_data.Cmx, force_data.Cmy, force_data.Cmz,
                u_inlet)
    end
end
