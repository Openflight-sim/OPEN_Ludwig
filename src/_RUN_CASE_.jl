// # FILE: ./src/_RUN_CASE_.jl
ENV["LC_ALL"] = "C" # CRITICAL FIX: Force standard decimal formatting (dots not commas)

include("initialize.jl")
include("lattice.jl")

if !isfile(joinpath(@__DIR__, "geometry.jl"))
    error("geometry.jl not found in $(@__DIR__)")
end
include("geometry.jl")
using .Geometry

include("domain.jl")
include("physics.jl")
include("diagnostics.jl")
include("forces.jl")
include("diagnostics_vram.jl")

using LinearAlgebra
using Printf
using WriteVTK
using Dates 
using StaticArrays
using KernelAbstractions
using CUDA
using Adapt
using Base.Threads
using YAML

const SIMULATION_START_TIME = Ref(time())

function walltime_str()
    elapsed = time() - SIMULATION_START_TIME[]
    hours = floor(Int, elapsed / 3600)
    mins = floor(Int, (elapsed % 3600) / 60)
    secs = elapsed % 60
    return @sprintf("%02d:%02d:%05.2f", hours, mins, secs)
end

function log_walltime(message::String)
    println("[$(walltime_str())] $message")
end

function execute_timestep_batch!(grids, t_start::Int, batch_size::Int, u_curr::Float32,
                              cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                              domain_nx::Int, domain_ny::Int, domain_nz::Int,
                              wall_model_active::Bool, c_wale_val::Float32)
    backend = get_backend(grids[1].rho)
    
    for t_offset in 0:(batch_size-1)
        t = t_start + t_offset
        
        recursive_step!(grids, 1, t, nothing, nothing, u_curr,
                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                        domain_nx, domain_ny, domain_nz, wall_model_active, c_wale_val)
    end
    
    KernelAbstractions.synchronize(backend)
end

function recursive_step!(grids, current_lvl::Int, t_sub::Int, parent_f, parent_ptr, u_vel::Float32,
                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                        domain_nx::Int, domain_ny::Int, domain_nz::Int,
                        wall_model_active::Bool, c_wale_val::Float32)
    if current_lvl > length(grids)
        return
    end
    
    level = grids[current_lvl]
    
    if iseven(t_sub)
        f_in = level.f
        f_out = level.f_temp
        vel_in = level.vel
        vel_out = level.vel_temp
    else
        f_in = level.f_temp
        f_out = level.f
        vel_in = level.vel_temp
        vel_out = level.vel
    end
    
    perform_timestep!(level, parent_f, parent_ptr, f_out, f_in, vel_out, vel_in, u_vel,
                      cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                      domain_nx, domain_ny, domain_nz, wall_model_active, c_wale_val,
                      t_sub)
    
    if current_lvl < length(grids)
        recursive_step!(grids, current_lvl + 1, 2*t_sub, f_out, level.block_pointer, u_vel,
                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                        domain_nx, domain_ny, domain_nz, wall_model_active, c_wale_val)
                        
        recursive_step!(grids, current_lvl + 1, 2*t_sub + 1, f_out, level.block_pointer, u_vel,
                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                        domain_nx, domain_ny, domain_nz, wall_model_active, c_wale_val)
    end
end

function compute_diagnostics_fast(level::BlockLevel, u_in::Float32, timestep::Int)
    backend = get_backend(level.rho)
    
    if backend isa CUDABackend
        rho_flat = reshape(level.rho, :)
        
        if iseven(timestep)
            vel_flat = reshape(level.vel_temp, :, 3)
        else
            vel_flat = reshape(level.vel, :, 3)
        end
        
        obs_flat = reshape(level.obstacle, :)
        
        valid_mask = .!obs_flat
        
        total_mass = sum(rho_flat .* valid_mask)
        
        v2 = sum(abs2, vel_flat, dims=2)
        ke_arr = 0.5f0 .* rho_flat .* vec(v2) .* valid_mask
        total_ke = sum(ke_arr)
        
        max_v2 = maximum(vec(v2) .* valid_mask)
        u_max = sqrt(max_v2)
        
        min_rho = minimum(rho_flat)
        max_rho = maximum(rho_flat)
        
        return total_mass, total_ke, u_max, min_rho, max_rho
    else
        return 0.0, 0.0, 0.0, 0.0, 0.0
    end
end

function force_cleanup()
    if CUDA.functional()
        GC.gc(true)
        GC.gc(true)
        CUDA.reclaim()
        
        free_mem, total_mem = CUDA.memory_info()
        @printf("[Memory] VRAM: %.2f GB free / %.2f GB total\n", 
                free_mem/1024^3, total_mem/1024^3)
        return free_mem
    end
    return 0
end

function export_merged_mesh_sync(t_step, grids, out_dir, backend)
    log_walltime("VTK Export START for step $t_step")
    vtk_start_time = time()
    
    level_active_blocks = [Set{Tuple{Int,Int,Int}}() for _ in 1:length(grids)]
    for lvl in 1:length(grids)
        for b in grids[lvl].active_block_coords
            push!(level_active_blocks[lvl], b)
        end
    end
    
    all_valid_blocks = Vector{NamedTuple{(:lvl, :b_idx, :bx, :by, :bz), Tuple{Int, Int, Int, Int, Int}}}()
    
    for lvl in 1:length(grids)
        level = grids[lvl]
        next_lvl_blocks = (lvl < length(grids)) ? level_active_blocks[lvl+1] : nothing
        
        for (b_idx, (bx, by, bz)) in enumerate(level.active_block_coords)
            should_export = true
            if next_lvl_blocks !== nothing
                children_exist = 0
                for dbz in 0:1, dby in 0:1, dbx in 0:1
                    child_coord = (2*bx - 1 + dbx, 2*by - 1 + dby, 2*bz - 1 + dbz)
                    if child_coord in next_lvl_blocks
                        children_exist += 1
                    end
                end
                if children_exist == 8; should_export = false; end
            end
            if should_export
                push!(all_valid_blocks, (lvl=lvl, b_idx=b_idx, bx=bx, by=by, bz=bz))
            end
        end
    end
    
    if isempty(all_valid_blocks); return; end
    
    level_data = Dict{Int, NamedTuple}()
    for lvl in 1:length(grids)
        level = grids[lvl]
        rho_cpu = Array(level.rho)
        vel_cpu = iseven(t_step) ? Array(level.vel_temp) : Array(level.vel)
        obs_cpu = Array(level.obstacle)
        
        w_cpu = nothing
        if OUTPUT_VORTICITY
            w_gpu = compute_vorticity(level)
            KernelAbstractions.synchronize(backend)
            w_cpu = Array(w_gpu)
        end
        
        bouzidi_mask = nothing
        if level.bouzidi_enabled && level.n_boundary_cells > 0
            n_blocks = length(level.active_block_coords)
            bouzidi_mask = falses(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
            cell_block_cpu = Array(level.bouzidi_cell_block)
            cell_x_cpu = Array(level.bouzidi_cell_x)
            cell_y_cpu = Array(level.bouzidi_cell_y)
            cell_z_cpu = Array(level.bouzidi_cell_z)
            for i in 1:level.n_boundary_cells
                b = cell_block_cpu[i]; x = cell_x_cpu[i]; y = cell_y_cpu[i]; z = cell_z_cpu[i]
                bouzidi_mask[x, y, z, b] = true
            end
        end
        level_data[lvl] = (rho=rho_cpu, vel=vel_cpu, obs=obs_cpu, w=w_cpu, dx=Float32(level.dx), bouzidi=bouzidi_mask)
    end
    
    n_blocks_total = length(all_valid_blocks)
    n_points_per_block = (BLOCK_SIZE + 1)^3
    n_cells_per_block = BLOCK_SIZE^3
    total_points = n_blocks_total * n_points_per_block
    total_cells = n_blocks_total * n_cells_per_block
    
    points_matrix = Matrix{Float32}(undef, 3, total_points)
    cells_vec = Vector{MeshCell}(undef, total_cells)
    rho_arr = Vector{Float32}(undef, total_cells)
    vel_matrix = Matrix{Float32}(undef, 3, total_cells)
    w_arr = Vector{Float32}(undef, total_cells)
    obst_arr = Vector{UInt8}(undef, total_cells)
    level_arr = Vector{Int32}(undef, total_cells)
    bouzidi_arr = Vector{UInt8}(undef, total_cells)
    
    stride_y = BLOCK_SIZE + 1
    stride_z = (BLOCK_SIZE + 1)^2
    
    Threads.@threads for i in 1:n_blocks_total
        block_info = all_valid_blocks[i]
        lvl = block_info.lvl
        b_idx = block_info.b_idx
        bx, by, bz = block_info.bx, block_info.by, block_info.bz
        data = level_data[lvl]
        dx = data.dx
        
        point_offset_base = (i - 1) * n_points_per_block
        cell_offset_base = (i - 1) * n_cells_per_block
        off_gx = Float32((bx - 1) * BLOCK_SIZE); off_gy = Float32((by - 1) * BLOCK_SIZE); off_gz = Float32((bz - 1) * BLOCK_SIZE)
        
        for pz in 0:BLOCK_SIZE, py in 0:BLOCK_SIZE, px in 0:BLOCK_SIZE
            local_pidx = 1 + px + py*(BLOCK_SIZE+1) + pz*(BLOCK_SIZE+1)^2
            global_pidx = point_offset_base + local_pidx
            points_matrix[1, global_pidx] = (off_gx + px) * dx
            points_matrix[2, global_pidx] = (off_gy + py) * dx
            points_matrix[3, global_pidx] = (off_gz + pz) * dx
        end
        
        for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
            local_c_idx = x + (y-1)*BLOCK_SIZE + (z-1)*BLOCK_SIZE^2
            global_c_idx = cell_offset_base + local_c_idx
            
            p_x, p_y, p_z = x-1, y-1, z-1
            pt_base = point_offset_base + 1 + p_x + p_y*stride_y + p_z*stride_z
            ids = (pt_base, pt_base + 1, pt_base + stride_y, pt_base + stride_y + 1,
                   pt_base + stride_z, pt_base + stride_z + 1, pt_base + stride_z + stride_y, pt_base + stride_z + stride_y + 1)
            cells_vec[global_c_idx] = MeshCell(VTKCellTypes.VTK_VOXEL, ids)
            
            vel_matrix[1, global_c_idx] = data.vel[x, y, z, b_idx, 1]
            vel_matrix[2, global_c_idx] = data.vel[x, y, z, b_idx, 2]
            vel_matrix[3, global_c_idx] = data.vel[x, y, z, b_idx, 3]
            rho_arr[global_c_idx] = data.rho[x, y, z, b_idx]
            if OUTPUT_VORTICITY && data.w !== nothing; w_arr[global_c_idx] = data.w[x, y, z, b_idx]; end
            obst_arr[global_c_idx] = data.obs[x, y, z, b_idx] ? 0x01 : 0x00
            level_arr[global_c_idx] = Int32(lvl)
            bouzidi_arr[global_c_idx] = (data.bouzidi !== nothing && data.bouzidi[x, y, z, b_idx]) ? 0x01 : 0x00
        end
    end
    
    # --------------------------------------------------------------------------
    # SANITIZE OUTPUT FOR STABILITY
    # Replace NaNs/Infs to prevent file corruption
    # --------------------------------------------------------------------------
    replace!(rho_arr, NaN32 => 0.0f0, Inf32 => 0.0f0, -Inf32 => 0.0f0)
    replace!(vel_matrix, NaN32 => 0.0f0, Inf32 => 0.0f0, -Inf32 => 0.0f0)
    if OUTPUT_VORTICITY; replace!(w_arr, NaN32 => 0.0f0, Inf32 => 0.0f0, -Inf32 => 0.0f0); end
    # --------------------------------------------------------------------------

    filename = @sprintf("%s/flow_%06d", out_dir, t_step)
    
    try
        # CRITICAL FIX: append=false ensures Base64 encoding which is safe against
        # Linux/Windows binary offset issues and locale corruptions.
        vtk_grid(filename, points_matrix, cells_vec; compress=true, append=false) do vtk
            if OUTPUT_DENSITY; vtk["Density"] = rho_arr; end
            if OUTPUT_VELOCITY; vtk["Velocity"] = vel_matrix; end
            if OUTPUT_VEL_MAG; vtk["VelocityMagnitude"] = sqrt.(vel_matrix[1,:].^2 .+ vel_matrix[2,:].^2 .+ vel_matrix[3,:].^2); end
            if OUTPUT_VORTICITY; vtk["Vorticity"] = w_arr; end
            if OUTPUT_OBSTACLE; vtk["Obstacle"] = obst_arr; end
            if OUTPUT_LEVEL; vtk["Level"] = level_arr; end
            if OUTPUT_BOUZIDI; vtk["BouzidiBoundary"] = bouzidi_arr; end
        end
    catch e
        println("[Error] Failed to write VTK file: $e")
    end
    
    vtk_elapsed = time() - vtk_start_time
    log_walltime(@sprintf("VTK Export END - %s.vtu (%.2fM cells, %.1fs)", filename, total_cells/1e6, vtk_elapsed))
end

function solve_main()
    SIMULATION_START_TIME[] = time()
    
    println("\n==========================================================")
    println("      LBM SOLVER | D3Q27 | WALE LES | A-B PATTERN        ")
    println("      Case: $(basename(CASE_DIR)) | $(Dates.now())          ")
    println("==========================================================")
    
    
    c_wale_current = Float32(C_SMAGO) 
    if haskey(CFG["advanced"]["numerics"], "c_wale")
        c_wale_current = Float32(CFG["advanced"]["numerics"]["c_wale"])
    end
    
    println("[Config] Fluid: ρ=$(FLUID_DENSITY) kg/m³, ν=$(FLUID_KINEMATIC_VISCOSITY) m²/s")
    println("[Config] Flow: U=$(FLOW_VELOCITY) m/s")
    println("[Config] Model: WALE Turbulence Model (C_w = $c_wale_current)")
    
    if BOUNDARY_METHOD == :bouzidi
        println("[Config] Boundary: Bouzidi Interpolated (2nd order, sparse)")
        println("[Config] Bouzidi levels: finest $BOUZIDI_LEVELS level(s)")
    else
        println("[Config] Boundary: Simple Bounce-Back (1st order)")
    end
    
    if FORCE_COMPUTATION_ENABLED
        println("[Config] Force/Moment computation: ENABLED")
    end
    
    println("[Init] Cleaning up previous resources...")
    free_bytes = force_cleanup()

    if CUDA.functional()
        println("[Backend] CUDA GPU: $(CUDA.name(CUDA.device()))")
        backend = CUDABackend()
        CUDA.allowscalar(false)
    else
        println("[Backend] No GPU detected. Using CPU.")
        backend = CPU()
    end

    
    output_dir = OUT_DIR 
    if isdir(output_dir)
        println("[Init] Cleaning existing output directory: $output_dir")
        for file in readdir(output_dir)
            rm(joinpath(output_dir, file); recursive=true, force=true)
        end
    else
        mkdir(output_dir)
    end
    
    csv_path = joinpath(output_dir, "convergence.csv")
    open(csv_path, "w") do io
        println(io, "Step,Walltime,Time_phys_s,U_inlet_lat,U_max_lat,Rho_min,Rho_max,MLUPS")
    end
    
    force_csv_path = joinpath(output_dir, "forces.csv")
    if FORCE_COMPUTATION_ENABLED
        write_force_csv_header(force_csv_path)
    end

    stl_path = STL_FILE
    if !isfile(stl_path)
        stl_simple = joinpath(CASE_DIR, "model.stl")
        if isfile(stl_simple)
            stl_path = stl_simple
        else
            println("[Error] STL file not found at: $stl_path")
            return
        end
    end
    
    log_walltime("Building domain...")
    cpu_grids = setup_multilevel_domain(stl_path; num_levels=NUM_LEVELS_CONFIG)
    
    params = get_domain_params()
    domain_nx = params.nx_coarse
    domain_ny = params.ny_coarse
    domain_nz = params.nz_coarse
    
    println("[Info] Coarse domain: $(domain_nx) × $(domain_ny) × $(domain_nz)")
    println("[Info] Reynolds number: $(round(params.re_number, digits=0))")
    println("[Info] Wall Model: $(params.wall_model_active ? "ENABLED" : "DISABLED")")
    
    log_walltime("Transferring to GPU...")
    grids = Vector{BlockLevel}()
    for g in cpu_grids
        push!(grids, adapt(backend, g))
    end
    cpu_grids = nothing
    GC.gc()
    
    log_walltime("Setting equilibrium distribution (A-B pattern)...")
    for lvl in 1:length(grids)
        level = grids[lvl]
        n_blks = length(level.active_block_coords)
        if n_blks > 0
            @kernel function init_eq_ab!(f, f_temp, W_vec)
                x, y, z, b = @index(Global, NTuple)
                @inbounds for k in 1:27
                    val = W_vec[k]
                    f[x, y, z, b, k] = val
                    f_temp[x, y, z, b, k] = val
                end
            end
            w_gpu = adapt(backend, W)
            k_init = init_eq_ab!(backend)
            k_init(level.f, level.f_temp, w_gpu, ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blks))
        end
    end
    KernelAbstractions.synchronize(backend)
    
    log_walltime("Building lattice arrays on GPU...")
    cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu = build_lattice_arrays_gpu(backend)
    
    total_cells = sum(length(g.active_block_coords) * BLOCK_SIZE^3 for g in grids)
    println("[Info] Total cells: $(round(total_cells/1e6, digits=2)) M")
    
    print_vram_breakdown(grids)
    
    force_data = nothing
    finest_level_idx = length(grids)
    if FORCE_COMPUTATION_ENABLED && grids[finest_level_idx].bouzidi_enabled
        force_data = ForceData(
            rho_ref = params.rho_physical,
            u_ref = params.u_physical,
            area_ref = params.reference_area,
            chord_ref = params.reference_chord,
            moment_center = params.moment_center,
            force_scale = params.force_scale,
            length_scale = params.length_scale,
            symmetric = SYMMETRIC_ANALYSIS
        )
        log_walltime(@sprintf("Force computation initialized (A=%.4f m², c=%.4f m)", 
                              params.reference_area, params.reference_chord))
    end
    
    log_walltime("LBM Analysis STARTED")
    @printf("%8s | %12s | %10s | %7s | %7s | %6s | %8s | %8s | %8s\n", 
            "Step", "Walltime", "Time[s]", "U_lat", "ρ_min", "MLUPS", "Cd", "Cl", "Cm")
    println(repeat("-", 100))

    t_start_simulation = time()
    last_diag_time = time()
    batch_size = GPU_ASYNC_DEPTH
    
    t = 1
    while t <= STEPS
        batch_end = min(t + batch_size - 1, STEPS)
        actual_batch = batch_end - t + 1
        
        prog = (batch_end <= RAMP_STEPS) ? 
               0.5f0 * (1.0f0 - Float32(cos(pi * batch_end / RAMP_STEPS))) : 1.0f0
        u_curr = U_TARGET * prog
        
        execute_timestep_batch!(grids, t, actual_batch, u_curr,
                               cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                               domain_nx, domain_ny, domain_nz,
                               params.wall_model_active, c_wale_current)
        
        if batch_end % DIAG_FREQ < actual_batch || batch_end == STEPS
            diag_step = (batch_end ÷ DIAG_FREQ) * DIAG_FREQ
            if diag_step >= t && diag_step <= batch_end
                l1 = grids[1]
                mass, ke, u_max, r_min, r_max = compute_diagnostics_fast(l1, u_curr, diag_step)
                
                curr_time = time()
                step_time = curr_time - last_diag_time
                last_diag_time = curr_time
                mlups = (total_cells * DIAG_FREQ) / (step_time * 1e6)
                time_phys = Float64(diag_step) * params.time_scale
                
                cd_str = "N/A"; cl_str = "N/A"; cm_str = "N/A"
                if force_data !== nothing
                    finest = grids[finest_level_idx]
                    f_curr = iseven(batch_end) ? finest.f : finest.f_temp
                    compute_forces!(force_data, finest, f_curr, cx_gpu, cy_gpu, cz_gpu, opp_gpu, Q_MIN_THRESHOLD, backend)
                    cd_str = @sprintf("%.4f", force_data.Cd)
                    cl_str = @sprintf("%.4f", force_data.Cl)
                    cm_str = @sprintf("%.4f", force_data.Cmy)
                    append_force_csv(force_csv_path, diag_step, time_phys, force_data, u_curr)
                end
                
                @printf("%8d | %12s | %10.4f | %.4f | %.4f | %6.1f | %8s | %8s | %8s\n", 
                        diag_step, walltime_str(), time_phys, u_curr, r_min, mlups, cd_str, cl_str, cm_str)
                open(csv_path, "a") do io
                    println(io, "$diag_step,$(walltime_str()),$time_phys,$u_curr,$u_max,$r_min,$r_max,$mlups")
                end
            end
        end
        
        if batch_end % OUTPUT_FREQ < actual_batch
            output_step = (batch_end ÷ OUTPUT_FREQ) * OUTPUT_FREQ
            if output_step >= t && output_step <= batch_end
                export_merged_mesh_sync(output_step, grids, output_dir, backend)
            end
        end
        
        t = batch_end + 1
    end
    
    total_time = time() - t_start_simulation
    avg_mlups = (total_cells * STEPS) / (total_time * 1e6)
    total_phys_time = Float64(STEPS) * params.time_scale
    
    println("\n==========================================================")
    println("    SIMULATION COMPLETE (WALE + A-B PATTERN)")
    @printf("    Wall time:     %.1f s\n", total_time)
    @printf("    Performance:   %.1f MLUPS\n", avg_mlups)
    println("==========================================================")

    grids = nothing
    force_cleanup()
end


function run_all_cases()
    cases_file = joinpath(@__DIR__, "../cases_to_run.yaml")
    if !isfile(cases_file)
        error("cases_to_run.yaml not found!")
    end
    
    config_yaml = YAML.load_file(cases_file)
    case_folders = config_yaml["case_folders"]
    
    println("==========================================================")
    println("      MULTI-CASE EXECUTION STARTED                        ")
    println("      Found $(length(case_folders)) cases to run.")
    println("==========================================================")
    
    for (i, case_name) in enumerate(case_folders)
        println("\n>>> STARTING CASE $i/$(length(case_folders)): $case_name")
        try
            
            load_case_configuration(case_name)
            
            
            solve_main()
            
            println(">>> FINISHED CASE $case_name\n")
        catch e
            println("!!! ERROR running case $case_name:")
            showerror(stdout, e, catch_backtrace())
            println("\n!!! Skipping to next case...")
        end
        
        
        GC.gc(true)
        if CUDA.functional(); CUDA.reclaim(); end
    end
    
    println("\n==========================================================")
    println("      ALL CASES COMPLETED                                 ")
    println("==========================================================")
end


run_all_cases()