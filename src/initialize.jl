# FILE: .\src\initialize.jl

"""
INITIALIZE.JL - Configuration Loader & Domain Setup

Refactored to allow sequential case execution by using global variables
instead of constants for case-specific parameters.
"""

using YAML
using Printf

# ==============================================================================
# 1. GLOBAL STATE CONTAINERS (Mutable)
# ==============================================================================

# Helper for safe dictionary access
function safe_get(dict, keys...; default=nothing)
    current = dict
    for (i, key) in enumerate(keys)
        if current === nothing || !isa(current, Dict) || !haskey(current, key)
            if default !== nothing
                return default
            end
            path = join(keys[1:i], " → ")
            error("Config Error: Missing key path '$path' in config.yaml")
        end
        current = current[key]
    end
    if current === nothing && default !== nothing
        return default
    end
    return current
end

# Global Configuration Dictionary
global CFG = Dict()
global CASE_DIR = ""

# --- Basic Parameters ---
global STL_FILENAME = ""
global STL_FILE = ""
global STL_SCALE = 1.0
global OUT_DIR_NAME = "RESULTS"
global OUT_DIR = ""
global SURFACE_RESOLUTION = 200
global NUM_LEVELS_CONFIG = 0

# --- Reference Values ---
global SYMMETRIC_ANALYSIS = false
global REFERENCE_AREA_FULL_MODEL = 0.0
global REFERENCE_AREA_CONFIG = 0.0
global REFERENCE_CHORD_CONFIG = 0.0
global REFERENCE_LENGTH_FOR_MESHING = 0.0
global REFERENCE_DIMENSION = :x

# --- Physics ---
global FLUID_DENSITY = 1.225
global FLUID_KINEMATIC_VISCOSITY = 1.5e-5
global FLOW_VELOCITY = 10.0

# --- Simulation ---
global STEPS = 1000
global RAMP_STEPS = 100
global OUTPUT_FREQ = 100

# --- Output Control ---
global OUTPUT_DENSITY = true
global OUTPUT_VELOCITY = true
global OUTPUT_VEL_MAG = true
global OUTPUT_VORTICITY = true
global OUTPUT_OBSTACLE = true
global OUTPUT_LEVEL = true
global OUTPUT_BOUZIDI = true

# --- Numerics ---
global U_TARGET = 0.05f0
global C_SMAGO = 0.1f0
global TAU_MIN = 0.505f0
global TAU_SAFETY_FACTOR = 1.1f0

# --- High Re ---
global AUTO_LEVELS = false
global MAX_LEVELS = 12
global MIN_COARSE_BLOCKS = 4
global WALL_MODEL_ENABLED = false
global WALL_MODEL_TYPE = :equilibrium
global WALL_MODEL_YPLUS_TARGET = 30.0

# --- Domain ---
global DOMAIN_UPSTREAM = 0.75
global DOMAIN_DOWNSTREAM = 1.5
global DOMAIN_LATERAL = 0.75
global DOMAIN_HEIGHT = 0.75
global SPONGE_THICKNESS = 0.10f0

# --- Refinement ---
# Note: BLOCK_SIZE is kept as const in blocks.jl, but we read it here for checks
global BLOCK_SIZE_CONFIG = 8 
global REFINEMENT_MARGIN = 2
global REFINEMENT_STRATEGY = :geometry_first
global ENABLE_WAKE_REFINEMENT = false
global WAKE_REFINEMENT_LENGTH = 0.25
global WAKE_REFINEMENT_WIDTH_FACTOR = 0.1
global WAKE_REFINEMENT_HEIGHT_FACTOR = 0.1

# --- Boundary ---
global BOUNDARY_METHOD = :bounce_back
global BOUZIDI_LEVELS = 1
global Q_MIN_THRESHOLD = 0.001f0

# --- Forces ---
global FORCE_COMPUTATION_ENABLED = false
global FORCE_OUTPUT_FREQ_CONFIG = 0
global FORCE_OUTPUT_FREQ = 0
global MOMENT_CENTER_CONFIG = [0.0, 0.0, 0.0]

# --- Diagnostics & GPU ---
global DIAG_FREQ = 100
global STABILITY_CHECK_ENABLED = true
global PRINT_TAU_WARNING = true
global GPU_ASYNC_DEPTH = 3
global USE_STREAMS = true
global PREFETCH_NEIGHBORS = true

# --- Computed Global Constants (Physics) ---
const CS2 = 1.0f0 / 3.0f0
const CS4 = CS2 * CS2

# --- Domain Parameters Struct ---
mutable struct DomainParameters
    initialized::Bool
    num_levels::Int  
    
    mesh_min::Tuple{Float64, Float64, Float64}
    mesh_max::Tuple{Float64, Float64, Float64}
    mesh_center::Tuple{Float64, Float64, Float64}
    mesh_extent::Tuple{Float64, Float64, Float64}
    
    reference_length::Float64
    reference_chord::Float64
    reference_area::Float64
    moment_center::Tuple{Float64, Float64, Float64}
    
    domain_min::Tuple{Float64, Float64, Float64}
    domain_max::Tuple{Float64, Float64, Float64}
    domain_size::Tuple{Float64, Float64, Float64}
    mesh_offset::Tuple{Float64, Float64, Float64}
    
    dx_fine::Float64
    dx_coarse::Float64
    dx_levels::Vector{Float64}
    
    nx_coarse::Int
    ny_coarse::Int
    nz_coarse::Int
    bx_max::Int
    by_max::Int
    bz_max::Int
    
    l_char::Float64
    nu_lattice::Float64
    tau_coarse::Float32
    tau_levels::Vector{Float32}
    cs2::Float32
    cs4::Float32
    
    re_number::Float64
    u_physical::Float64
    rho_physical::Float64
    nu_physical::Float64
    
    length_scale::Float64
    time_scale::Float64
    velocity_scale::Float64
    force_scale::Float64
    
    tau_fine::Float64
    tau_margin_percent::Float64
    wall_model_active::Bool
    y_plus_first_cell::Float64
    
    estimated_memory_gb::Float64
end

# Global instance of DomainParameters
global DOMAIN_PARAMS = DomainParameters(
    false, 0,
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    0.0, 0.0, 0.0, (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    0.0, 0.0, Float64[],
    0, 0, 0, 0, 0, 0,
    0.0, 0.0, 0.0f0, Float32[], 1.0f0/3.0f0, 1.0f0/9.0f0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, false, 0.0,
    0.0
)

# ==============================================================================
# 2. CONFIGURATION LOADING FUNCTION
# ==============================================================================

"""
    load_case_configuration(case_folder_name)

Reads config.yaml from the specified case folder and populates global variables.
"""
function load_case_configuration(case_folder_name::String)
    global CASE_DIR = abspath(joinpath(@__DIR__, "../CASES", case_folder_name))
    
    if !isdir(CASE_DIR)
        error("CRITICAL: Case folder not found at $CASE_DIR")
    end
    
    config_path = joinpath(CASE_DIR, "config.yaml")
    if !isfile(config_path)
        error("CRITICAL: config.yaml not found at $config_path")
    end
    
    println("[Init] Loading case configuration: $case_folder_name")
    
    # Update global CFG dictionary
    global CFG = YAML.load_file(config_path)
    
    # --- Update Globals from CFG ---
    
    global STL_FILENAME = safe_get(CFG, "basic", "stl_file")
    global STL_FILE = joinpath(CASE_DIR, STL_FILENAME)
    global STL_SCALE = Float64(safe_get(CFG, "basic", "stl_scale"))
    
    global OUT_DIR_NAME = safe_get(CFG, "basic", "simulation", "output_dir")
    global OUT_DIR = joinpath(CASE_DIR, OUT_DIR_NAME)
    
    global SURFACE_RESOLUTION = Int(safe_get(CFG, "basic", "surface_resolution"))
    global NUM_LEVELS_CONFIG = Int(safe_get(CFG, "basic", "num_levels"))
    
    global SYMMETRIC_ANALYSIS = safe_get(CFG, "advanced", "refinement", "symmetric_analysis"; default=false)
    global REFERENCE_AREA_FULL_MODEL = Float64(safe_get(CFG, "basic", "reference_area_of_full_model"; default=0.0))
    global REFERENCE_AREA_CONFIG = SYMMETRIC_ANALYSIS ? REFERENCE_AREA_FULL_MODEL / 2.0 : REFERENCE_AREA_FULL_MODEL
    
    global REFERENCE_CHORD_CONFIG = Float64(safe_get(CFG, "basic", "reference_chord"; default=0.0))
    global REFERENCE_LENGTH_FOR_MESHING = Float64(safe_get(CFG, "basic", "reference_length_for_meshing"; default=0.0))
    global REFERENCE_DIMENSION = Symbol(safe_get(CFG, "basic", "reference_dimension"; default="x"))
    
    global FLUID_DENSITY = Float64(safe_get(CFG, "basic", "fluid", "density"; default=1.225))
    global FLUID_KINEMATIC_VISCOSITY = Float64(safe_get(CFG, "basic", "fluid", "kinematic_viscosity"; default=1.5e-5))
    global FLOW_VELOCITY = Float64(safe_get(CFG, "basic", "flow", "velocity"; default=10.0))
    
    global STEPS = Int(safe_get(CFG, "basic", "simulation", "steps"))
    global RAMP_STEPS = Int(safe_get(CFG, "basic", "simulation", "ramp_steps"))
    global OUTPUT_FREQ = Int(safe_get(CFG, "basic", "simulation", "output_freq"))
    
    global OUTPUT_DENSITY = safe_get(CFG, "basic", "simulation", "output_fields", "density"; default=true)
    global OUTPUT_VELOCITY = safe_get(CFG, "basic", "simulation", "output_fields", "velocity"; default=true)
    global OUTPUT_VEL_MAG = safe_get(CFG, "basic", "simulation", "output_fields", "velocity_magnitude"; default=true)
    global OUTPUT_VORTICITY = safe_get(CFG, "basic", "simulation", "output_fields", "vorticity"; default=true)
    global OUTPUT_OBSTACLE = safe_get(CFG, "basic", "simulation", "output_fields", "obstacle"; default=true)
    global OUTPUT_LEVEL = safe_get(CFG, "basic", "simulation", "output_fields", "level"; default=true)
    global OUTPUT_BOUZIDI = safe_get(CFG, "basic", "simulation", "output_fields", "bouzidi"; default=true)
    
    global U_TARGET = Float32(safe_get(CFG, "advanced", "numerics", "u_lattice"; default=0.01))
    global C_SMAGO = Float32(safe_get(CFG, "advanced", "numerics", "c_smago"; default=0.1))
    global TAU_MIN = Float32(safe_get(CFG, "advanced", "numerics", "tau_min"; default=0.505))
    global TAU_SAFETY_FACTOR = Float32(safe_get(CFG, "advanced", "numerics", "tau_safety_factor"; default=1.0))
    
    global AUTO_LEVELS = safe_get(CFG, "advanced", "high_re", "auto_levels"; default=false)
    global MAX_LEVELS = Int(safe_get(CFG, "advanced", "high_re", "max_levels"; default=12))
    global MIN_COARSE_BLOCKS = Int(safe_get(CFG, "advanced", "high_re", "min_coarse_blocks"; default=4))
    
    global WALL_MODEL_ENABLED = safe_get(CFG, "advanced", "high_re", "wall_model", "enabled"; default=false)
    global WALL_MODEL_TYPE = Symbol(safe_get(CFG, "advanced", "high_re", "wall_model", "type"; default="equilibrium"))
    global WALL_MODEL_YPLUS_TARGET = Float64(safe_get(CFG, "advanced", "high_re", "wall_model", "y_plus_target"; default=30.0))
    
    global DOMAIN_UPSTREAM = Float64(safe_get(CFG, "advanced", "domain", "upstream"; default=0.75))
    global DOMAIN_DOWNSTREAM = Float64(safe_get(CFG, "advanced", "domain", "downstream"; default=1.5))
    global DOMAIN_LATERAL = Float64(safe_get(CFG, "advanced", "domain", "lateral"; default=0.75))
    global DOMAIN_HEIGHT = Float64(safe_get(CFG, "advanced", "domain", "height"; default=0.75))
    global SPONGE_THICKNESS = Float32(safe_get(CFG, "advanced", "domain", "sponge_thickness"; default=0.10))
    
    global BLOCK_SIZE_CONFIG = Int(safe_get(CFG, "advanced", "refinement", "block_size"; default=8))
    global REFINEMENT_MARGIN = Int(safe_get(CFG, "advanced", "refinement", "margin"; default=2))
    global REFINEMENT_STRATEGY = Symbol(safe_get(CFG, "advanced", "refinement", "strategy"; default="geometry_first"))
    
    global ENABLE_WAKE_REFINEMENT = safe_get(CFG, "advanced", "refinement", "wake_enabled"; default=false)
    global WAKE_REFINEMENT_LENGTH = Float64(safe_get(CFG, "advanced", "refinement", "wake_length"; default=0.25))
    global WAKE_REFINEMENT_WIDTH_FACTOR = Float64(safe_get(CFG, "advanced", "refinement", "wake_width_factor"; default=0.1))
    global WAKE_REFINEMENT_HEIGHT_FACTOR = Float64(safe_get(CFG, "advanced", "refinement", "wake_height_factor"; default=0.1))
    
    global BOUNDARY_METHOD = Symbol(safe_get(CFG, "advanced", "boundary", "method"; default="bouzidi"))
    global BOUZIDI_LEVELS = Int(safe_get(CFG, "advanced", "boundary", "bouzidi_levels"; default=1))
    global Q_MIN_THRESHOLD = Float32(safe_get(CFG, "advanced", "boundary", "q_min_threshold"; default=0.001))
    
    global FORCE_COMPUTATION_ENABLED = safe_get(CFG, "advanced", "forces", "enabled"; default=true)
    global FORCE_OUTPUT_FREQ_CONFIG = Int(safe_get(CFG, "advanced", "forces", "output_freq"; default=0))
    global MOMENT_CENTER_CONFIG = safe_get(CFG, "advanced", "forces", "moment_center"; default=[0.25, 0.0, 0.0])
    global FORCE_OUTPUT_FREQ = FORCE_OUTPUT_FREQ_CONFIG == 0 ? DIAG_FREQ : FORCE_OUTPUT_FREQ_CONFIG
    
    global DIAG_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "freq"; default=500))
    global STABILITY_CHECK_ENABLED = safe_get(CFG, "advanced", "diagnostics", "stability_check"; default=true)
    global PRINT_TAU_WARNING = safe_get(CFG, "advanced", "diagnostics", "print_tau_warning"; default=true)
    
    global GPU_ASYNC_DEPTH = Int(safe_get(CFG, "advanced", "gpu", "async_depth"; default=8))
    global USE_STREAMS = safe_get(CFG, "advanced", "gpu", "use_streams"; default=true)
    global PREFETCH_NEIGHBORS = safe_get(CFG, "advanced", "gpu", "prefetch_neighbors"; default=true)
    
    # Reset Domain Params
    DOMAIN_PARAMS.initialized = false
    
    println("[Init] Configuration loaded successfully.")
end

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

function compute_tau_for_levels(Re::Float64, ref_length::Float64, resolution::Int, 
                                n_levels::Int, u_lattice::Float32)
    nu_lattice_fine = Float64(u_lattice) * resolution / Re
    tau_fine = 3.0 * nu_lattice_fine + 0.5
    return tau_fine
end

function compute_max_levels_for_domain(domain_size::Float64, dx_fine::Float64, 
                                       block_size::Int, min_blocks::Int)
    ratio = domain_size / (dx_fine * min_blocks * block_size)
    if ratio < 1.0
        return 1
    end
    return Int(floor(1 + log2(ratio)))
end

function compute_required_resolution_for_tau(Re::Float64, tau_target::Float64, u_lattice::Float32)
    resolution = (tau_target - 0.5) * Re / (3.0 * Float64(u_lattice))
    return Int(ceil(resolution))
end

function print_re_analysis(Re::Float64, ref_length::Float64, resolution::Int, u_lattice::Float32)
    tau = compute_tau_for_levels(Re, ref_length, resolution, 1, u_lattice)
    
    println("\n[High-Re] ═══════════════════════════════════════════════════════")
    @printf("[High-Re] Reynolds number: %.2e\n", Re)
    @printf("[High-Re] Reference length: %.4f m\n", ref_length)
    @printf("[High-Re] Surface resolution: %d cells/L\n", resolution)
    @printf("[High-Re] Lattice velocity: %.3f\n", u_lattice)
    println("[High-Re] ───────────────────────────────────────────────────────")
    @printf("[High-Re] Computed τ (finest): %.6f\n", tau)
    
    if tau < 0.500001
        println("[High-Re] ⚠ CRITICAL: τ ≈ 0.5 → UNSTABLE (zero viscosity)")
    elseif tau < 0.51
        println("[High-Re] ⚠ WARNING: τ very close to stability limit")
    else
        println("[High-Re] ✓ τ is in safe range")
    end
    println("[High-Re] ═══════════════════════════════════════════════════════\n")
end

function compute_domain_from_mesh(mesh_min::Tuple{Float64,Float64,Float64}, 
                                  mesh_max::Tuple{Float64,Float64,Float64})
    
    mesh_center = ((mesh_min[1] + mesh_max[1])/2,
                   (mesh_min[2] + mesh_max[2])/2,
                   (mesh_min[3] + mesh_max[3])/2)
    mesh_extent = (mesh_max[1] - mesh_min[1],
                   mesh_max[2] - mesh_min[2],
                   mesh_max[3] - mesh_min[3])
    
    if REFERENCE_LENGTH_FOR_MESHING > 0.0
        ref_length = REFERENCE_LENGTH_FOR_MESHING
    else
        ref_length = REFERENCE_DIMENSION == :x ? mesh_extent[1] :
                     REFERENCE_DIMENSION == :y ? mesh_extent[2] :
                     REFERENCE_DIMENSION == :z ? mesh_extent[3] : maximum(mesh_extent)
    end
    
    ref_chord = REFERENCE_CHORD_CONFIG > 0.0 ? REFERENCE_CHORD_CONFIG : mesh_extent[1]
    
    if REFERENCE_AREA_CONFIG > 0.0
        ref_area = REFERENCE_AREA_CONFIG
    else
        ref_area = mesh_extent[2] * mesh_extent[3]
        if SYMMETRIC_ANALYSIS
            ref_area *= 2.0  
        end
    end
    
    moment_center_rel = (Float64(MOMENT_CENTER_CONFIG[1]), 
                         Float64(MOMENT_CENTER_CONFIG[2]), 
                         Float64(MOMENT_CENTER_CONFIG[3]))
    
    u_phys = FLOW_VELOCITY
    nu_phys = FLUID_KINEMATIC_VISCOSITY
    rho_phys = FLUID_DENSITY
    re_number = u_phys * ref_length / nu_phys
    
    tau_fine_computed = compute_tau_for_levels(re_number, ref_length, SURFACE_RESOLUTION, 1, U_TARGET)
    
    if tau_fine_computed < TAU_MIN
        tau_fine = TAU_MIN
    else
        tau_fine = tau_fine_computed
    end
    
    print_re_analysis(re_number, ref_length, SURFACE_RESOLUTION, U_TARGET)
    
    domain_x = ref_length * (DOMAIN_UPSTREAM + DOMAIN_DOWNSTREAM) + mesh_extent[1]
    domain_y = SYMMETRIC_ANALYSIS ? (mesh_max[2] + ref_length * DOMAIN_LATERAL) :
                                    (mesh_extent[2] + 2.0 * ref_length * DOMAIN_LATERAL)
    domain_z = mesh_extent[3] + 2.0 * ref_length * DOMAIN_HEIGHT
    
    dx_fine = ref_length / SURFACE_RESOLUTION
    
    min_domain = min(domain_x, domain_y, domain_z)
    # Using BLOCK_SIZE_CONFIG which must match the const in blocks.jl
    max_levels_domain = compute_max_levels_for_domain(min_domain, dx_fine, BLOCK_SIZE_CONFIG, MIN_COARSE_BLOCKS)
    
    if NUM_LEVELS_CONFIG > 0
        num_levels = NUM_LEVELS_CONFIG
        if num_levels > max_levels_domain
            println("[Domain] ⚠ WARNING: Requested $num_levels levels exceeds domain limit $max_levels_domain")
            num_levels = max_levels_domain
        end
    elseif AUTO_LEVELS
        num_levels = min(max_levels_domain, MAX_LEVELS)
    else
        num_levels = min(8, max_levels_domain)
    end
    
    dx_coarse = dx_fine * 2^(num_levels - 1)
    dx_levels = [dx_fine * 2^(num_levels - lvl) for lvl in 1:num_levels]
    
    nx_coarse = max(BLOCK_SIZE_CONFIG, Int(ceil(ceil(domain_x / dx_coarse) / BLOCK_SIZE_CONFIG) * BLOCK_SIZE_CONFIG))
    ny_coarse = max(BLOCK_SIZE_CONFIG, Int(ceil(ceil(domain_y / dx_coarse) / BLOCK_SIZE_CONFIG) * BLOCK_SIZE_CONFIG))
    nz_coarse = max(BLOCK_SIZE_CONFIG, Int(ceil(ceil(domain_z / dx_coarse) / BLOCK_SIZE_CONFIG) * BLOCK_SIZE_CONFIG))
    
    domain_x = nx_coarse * dx_coarse
    domain_y = ny_coarse * dx_coarse
    domain_z = nz_coarse * dx_coarse
    
    bx_max = nx_coarse ÷ BLOCK_SIZE_CONFIG
    by_max = ny_coarse ÷ BLOCK_SIZE_CONFIG
    bz_max = nz_coarse ÷ BLOCK_SIZE_CONFIG
    
    mesh_x = ref_length * DOMAIN_UPSTREAM
    mesh_y = SYMMETRIC_ANALYSIS ? 0.0 : (domain_y / 2.0 - mesh_center[2])
    mesh_z = domain_z / 2.0 - mesh_center[3]
    mesh_offset = (mesh_x - mesh_min[1], mesh_y, mesh_z)
    
    length_scale = dx_fine                                  
    velocity_scale = u_phys / Float64(U_TARGET)        
    time_scale = length_scale / velocity_scale        
    
    nu_lattice_fine = nu_phys * time_scale / (length_scale^2)
    
    tau_levels = Float32[]
    for lvl in 1:num_levels
        if lvl == num_levels
            tau_lvl = tau_fine
        else
            scale_factor = 2.0^(num_levels - lvl)
            tau_lvl = 0.5 + (tau_fine - 0.5) * scale_factor
        end
        push!(tau_levels, Float32(tau_lvl))
    end
    
    tau_coarse = tau_levels[1]
    tau_margin_percent = (tau_fine - 0.5) / 0.5 * 100
    
    force_scale = rho_phys * length_scale^4 / time_scale^2
    
    moment_center_phys = (mesh_min[1] + mesh_offset[1] + moment_center_rel[1] * ref_chord, 
                          mesh_center[2] + mesh_offset[2] + moment_center_rel[2] * ref_chord, 
                          mesh_center[3] + mesh_offset[3] + moment_center_rel[3] * ref_chord)
    
    bytes_per_cell = 160
    total_cells_est = bx_max * by_max * bz_max * BLOCK_SIZE_CONFIG^3  
    for lvl in 2:num_levels
        total_cells_est += Int(ceil(total_cells_est * 0.08))
    end
    estimated_memory_gb = total_cells_est * bytes_per_cell / 1e9
    
    DOMAIN_PARAMS.initialized = true
    DOMAIN_PARAMS.num_levels = num_levels
    DOMAIN_PARAMS.mesh_min = mesh_min
    DOMAIN_PARAMS.mesh_max = mesh_max
    DOMAIN_PARAMS.mesh_center = mesh_center
    DOMAIN_PARAMS.mesh_extent = mesh_extent
    DOMAIN_PARAMS.reference_length = ref_length
    DOMAIN_PARAMS.reference_chord = ref_chord
    DOMAIN_PARAMS.reference_area = ref_area
    DOMAIN_PARAMS.moment_center = moment_center_phys
    DOMAIN_PARAMS.domain_min = (0.0, 0.0, 0.0)
    DOMAIN_PARAMS.domain_max = (domain_x, domain_y, domain_z)
    DOMAIN_PARAMS.domain_size = (domain_x, domain_y, domain_z)
    DOMAIN_PARAMS.mesh_offset = mesh_offset
    DOMAIN_PARAMS.dx_fine = dx_fine
    DOMAIN_PARAMS.dx_coarse = dx_coarse
    DOMAIN_PARAMS.dx_levels = dx_levels
    DOMAIN_PARAMS.nx_coarse = nx_coarse
    DOMAIN_PARAMS.ny_coarse = ny_coarse
    DOMAIN_PARAMS.nz_coarse = nz_coarse
    DOMAIN_PARAMS.bx_max = bx_max
    DOMAIN_PARAMS.by_max = by_max
    DOMAIN_PARAMS.bz_max = bz_max
    DOMAIN_PARAMS.l_char = ref_length / dx_coarse
    DOMAIN_PARAMS.nu_lattice = nu_lattice_fine
    DOMAIN_PARAMS.tau_coarse = tau_coarse
    DOMAIN_PARAMS.tau_levels = tau_levels
    DOMAIN_PARAMS.cs2 = 1.0f0/3.0f0
    DOMAIN_PARAMS.cs4 = 1.0f0/9.0f0
    DOMAIN_PARAMS.re_number = re_number
    DOMAIN_PARAMS.u_physical = u_phys
    DOMAIN_PARAMS.rho_physical = rho_phys
    DOMAIN_PARAMS.nu_physical = nu_phys
    DOMAIN_PARAMS.length_scale = length_scale
    DOMAIN_PARAMS.time_scale = time_scale
    DOMAIN_PARAMS.velocity_scale = velocity_scale
    DOMAIN_PARAMS.force_scale = force_scale
    DOMAIN_PARAMS.tau_fine = tau_fine
    DOMAIN_PARAMS.tau_margin_percent = tau_margin_percent
    DOMAIN_PARAMS.wall_model_active = WALL_MODEL_ENABLED
    DOMAIN_PARAMS.y_plus_first_cell = 0.0 # Placeholder
    DOMAIN_PARAMS.estimated_memory_gb = estimated_memory_gb
    
    return DOMAIN_PARAMS
end

function print_domain_summary()
    if !DOMAIN_PARAMS.initialized
        println("Domain not initialized.")
        return
    end
    p = DOMAIN_PARAMS
    
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║                LBM DOMAIN CONFIGURATION SUMMARY                  ║")
    println("╠══════════════════════════════════════════════════════════════════╣")
    @printf("║  Case:                %30s     ║\n", basename(CASE_DIR))
    @printf("║  Reynolds Number:     %12.0f                            ║\n", p.re_number)
    @printf("║  Resolution:          %12.6f m (finest)                 ║\n", p.dx_fine)
    @printf("║  Levels:              %12d                              ║\n", p.num_levels)
    @printf("║  Est. Memory:         %12.2f GB                           ║\n", p.estimated_memory_gb)
    println("╚══════════════════════════════════════════════════════════════════╝")
end

function get_domain_params()
    if !DOMAIN_PARAMS.initialized
        error("Domain not initialized. Call compute_domain_from_mesh() first.")
    end
    return DOMAIN_PARAMS
end

get_num_levels() = DOMAIN_PARAMS.initialized ? DOMAIN_PARAMS.num_levels : NUM_LEVELS_CONFIG

NUM_LEVELS = NUM_LEVELS_CONFIG
NUM_LEVELS_REQUESTED = NUM_LEVELS_CONFIG
RE_TARGET = 0.0

println("[Init] Initialization complete.")