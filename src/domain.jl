// # FILE: .\src\domain.jl
"""
DOMAIN.JL - Multi-Level Block Domain Generation
"""

include("blocks.jl")
include("bouzidi.jl")

using Base.Threads
using LinearAlgebra
using StaticArrays
using Printf


function triangle_intersects_aabb(center, box_half_size, v1, v2, v3)
    tol = 1.001
    h = box_half_size * tol
    t1 = v1 - center; t2 = v2 - center; t3 = v3 - center
    
    if min(t1[1], t2[1], t3[1]) > h[1] || max(t1[1], t2[1], t3[1]) < -h[1]; return false; end
    if min(t1[2], t2[2], t3[2]) > h[2] || max(t1[2], t2[2], t3[2]) < -h[2]; return false; end
    if min(t1[3], t2[3], t3[3]) > h[3] || max(t1[3], t2[3], t3[3]) < -h[3]; return false; end
    
    f = [t2 - t1, t3 - t2, t1 - t3]
    u = [SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0)]
    
    for i in 1:3
        for j in 1:3
            axis = cross(u[i], f[j])
            if dot(axis, axis) < 1e-10; continue; end
            p1 = dot(t1, axis); p2 = dot(t2, axis); p3 = dot(t3, axis)
            r = h[1]*abs(axis[1]) + h[2]*abs(axis[2]) + h[3]*abs(axis[3])
            if min(p1, min(p2, p3)) > r || max(p1, max(p2, p3)) < -r; return false; end
        end
    end
    return true
end

function get_active_blocks_for_level(mesh::Geometry.SolverMesh, 
                                     dx::Float64, 
                                     mesh_offset::SVector{3,Float64},
                                     bx_max::Int, by_max::Int, bz_max::Int)
    
    active_set = Set{Tuple{Int, Int, Int}}()
    bs = BLOCK_SIZE
    
    margin = dx * 0.01 
    inv_bs_dx = 1.0 / (bs * dx)
    
    for tri in mesh.triangles
        t_min = tri[1] .+ mesh_offset
        t_max = tri[1] .+ mesh_offset
        
        for p in tri
            pm = p .+ mesh_offset
            t_min = min.(t_min, pm)
            t_max = max.(t_max, pm)
        end
        
        min_bx = floor(Int, (t_min[1] - margin) * inv_bs_dx) + 1
        min_by = floor(Int, (t_min[2] - margin) * inv_bs_dx) + 1
        min_bz = floor(Int, (t_min[3] - margin) * inv_bs_dx) + 1
        
        max_bx = floor(Int, (t_max[1] + margin) * inv_bs_dx) + 1
        max_by = floor(Int, (t_max[2] + margin) * inv_bs_dx) + 1
        max_bz = floor(Int, (t_max[3] + margin) * inv_bs_dx) + 1
        
        min_bx = max(1, min_bx); max_bx = min(max_bx, bx_max)
        min_by = max(1, min_by); max_by = min(max_by, by_max)
        min_bz = max(1, min_bz); max_bz = min(max_bz, bz_max)
        
        for bz in min_bz:max_bz
            for by in min_by:max_by
                for bx in min_bx:max_bx
                    push!(active_set, (bx, by, bz))
                end
            end
        end
    end
    
    return active_set
end

function add_halo_blocks_with_siblings!(active_set::Set{Tuple{Int, Int, Int}}, 
                                        layers::Int,
                                        bx_max::Int, by_max::Int, bz_max::Int)
    neighbor_offsets = Tuple{Int,Int,Int}[]
    for dz in -1:1, dy in -1:1, dx in -1:1
        if dx==0 && dy==0 && dz==0; continue; end
        push!(neighbor_offsets, (dx, dy, dz))
    end
    
    for _ in 1:layers
        new_blocks = Set{Tuple{Int, Int, Int}}()
        
        for (bx, by, bz) in active_set
            for (dx, dy, dz) in neighbor_offsets
                nbx, nby, nbz = bx+dx, by+dy, bz+dz
                if nbx >= 1 && nbx <= bx_max && nby >= 1 && nby <= by_max && nbz >= 1 && nbz <= bz_max
                    if !((nbx, nby, nbz) in active_set)
                        push!(new_blocks, (nbx, nby, nbz))
                    end
                end
            end
        end
        
        siblings_to_add = Set{Tuple{Int, Int, Int}}()
        for (bx, by, bz) in new_blocks
            pbx = (bx + 1) ÷ 2
            pby = (by + 1) ÷ 2
            pbz = (bz + 1) ÷ 2
            
            for dbz in 0:1, dby in 0:1, dbx in 0:1
                sbx = 2*pbx - 1 + dbx
                sby = 2*pby - 1 + dby
                sbz = 2*pbz - 1 + dbz
                
                if sbx >= 1 && sbx <= bx_max && sby >= 1 && sby <= by_max && sbz >= 1 && sbz <= bz_max
                    if !((sbx, sby, sbz) in active_set) && !((sbx, sby, sbz) in new_blocks)
                        push!(siblings_to_add, (sbx, sby, sbz))
                    end
                end
            end
        end
        
        union!(active_set, new_blocks)
        union!(active_set, siblings_to_add)
    end
end

function ensure_complete_parent_coverage!(active_set::Set{Tuple{Int, Int, Int}},
                                          bx_max::Int, by_max::Int, bz_max::Int)
    added = true
    iterations = 0
    max_iterations = 10
    
    while added && iterations < max_iterations
        added = false
        iterations += 1
        siblings_to_add = Set{Tuple{Int, Int, Int}}()
        
        for (bx, by, bz) in active_set
            pbx = (bx + 1) ÷ 2
            pby = (by + 1) ÷ 2
            pbz = (bz + 1) ÷ 2
            
            for dbz in 0:1, dby in 0:1, dbx in 0:1
                sbx = 2*pbx - 1 + dbx
                sby = 2*pby - 1 + dby
                sbz = 2*pbz - 1 + dbz
                
                if sbx >= 1 && sbx <= bx_max && sby >= 1 && sby <= by_max && sbz >= 1 && sbz <= bz_max
                    if !((sbx, sby, sbz) in active_set)
                        push!(siblings_to_add, (sbx, sby, sbz))
                        added = true
                    end
                end
            end
        end
        
        union!(active_set, siblings_to_add)
    end
end

function build_neighbor_table(active_coords::Vector{Tuple{Int, Int, Int}}, 
                              bx_max::Int, by_max::Int, bz_max::Int)
    n_blocks = length(active_coords)
    table = zeros(Int32, n_blocks, 27)
    
    temp_ptr = zeros(Int32, bx_max, by_max, bz_max)
    for (i, (bx, by, bz)) in enumerate(active_coords)
        if bx >= 1 && bx <= bx_max && by >= 1 && by <= by_max && bz >= 1 && bz <= bz_max
            temp_ptr[bx, by, bz] = Int32(i)
        end
    end
    
    for i in 1:n_blocks
        (bx, by, bz) = active_coords[i]
        for dz in -1:1, dy in -1:1, dx in -1:1
            dir_idx = (dx+1) + (dy+1)*3 + (dz+1)*9 + 1
            nbx, nby, nbz = bx+dx, by+dy, bz+dz
            if nbx >= 1 && nbx <= bx_max && nby >= 1 && nby <= by_max && nbz >= 1 && nbz <= bz_max
                table[i, dir_idx] = temp_ptr[nbx, nby, nbz]
            else
                table[i, dir_idx] = 0
            end
        end
    end
    return table
end

function build_block_triangle_map(mesh::Geometry.SolverMesh, 
                                  sorted_blocks::Vector{Tuple{Int, Int, Int}}, 
                                  dx::Float64,
                                  mesh_offset::SVector{3,Float64})
    block_tris = [Int[] for _ in 1:length(sorted_blocks)]
    bs = BLOCK_SIZE
    
    b_lookup = Dict{Tuple{Int, Int, Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        b_lookup[coord] = i
    end
    
    margin = dx * 2
    
    for (t_idx, tri) in enumerate(mesh.triangles)
        min_pt = tri[1] .+ mesh_offset
        max_pt = tri[1] .+ mesh_offset
        for p in tri
             pm = p .+ mesh_offset
             min_pt = min.(min_pt, pm)
             max_pt = max.(max_pt, pm)
        end
        
        min_b = floor.(Int, (min_pt .- margin) ./ (bs * dx)) .+ 1
        max_b = floor.(Int, (max_pt .+ margin) ./ (bs * dx)) .+ 1
        
        for bz in max(1, min_b[3]):max_b[3]
            for by in max(1, min_b[2]):max_b[2]
                for bx in max(1, min_b[1]):max_b[1]
                    if haskey(b_lookup, (bx, by, bz))
                        idx = b_lookup[(bx, by, bz)]
                        push!(block_tris[idx], t_idx)
                    end
                end
            end
        end
    end
    return block_tris
end

function voxelize_blocks!(obstacle_arr, sorted_blocks, mesh::Geometry.SolverMesh, 
                          dx::Float64, mesh_offset::SVector{3,Float64})
    block_tri_map = build_block_triangle_map(mesh, sorted_blocks, dx, mesh_offset)
    
    println("[Domain] SAT Surface Marking...")
    box_half = SVector(0.75, 0.75, 0.75) * dx
    
    @threads for i in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[i]
        relevant_tris = block_tri_map[i]
        if isempty(relevant_tris)
            continue
        end
        
        for lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
            px = ((bx - 1) * BLOCK_SIZE + lx - 0.5) * dx
            py = ((by - 1) * BLOCK_SIZE + ly - 0.5) * dx
            pz = ((bz - 1) * BLOCK_SIZE + lz - 0.5) * dx
            center = SVector(px, py, pz)
            
            is_shell = false
            for tid in relevant_tris
                t = mesh.triangles[tid]
                v1 = t[1] .+ mesh_offset
                v2 = t[2] .+ mesh_offset
                v3 = t[3] .+ mesh_offset
                
                if triangle_intersects_aabb(center, box_half, v1, v2, v3)
                    is_shell = true
                    break
                end
            end
            
            if is_shell
                obstacle_arr[lx, ly, lz, i] = true
            end
        end
    end
end

function perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, 
                             grid_dim_x, grid_dim_y, grid_dim_z)
    println("[Domain] Flood Fill...")
    visited = falses(size(obstacle_arr))
    q_block = Int32[]
    q_idx = Int32[]
    
    min_x_block = minimum(b[1] for b in sorted_blocks)
    
    for b_idx in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[b_idx]
        if bx == min_x_block
            for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
                if !obstacle_arr[x, y, z, b_idx]
                    visited[x, y, z, b_idx] = true
                    push!(q_block, b_idx)
                    push!(q_idx, x + (y-1)*BLOCK_SIZE + (z-1)*BLOCK_SIZE*BLOCK_SIZE)
                end
            end
        end
    end
    
    head = 1
    dx_arr = [1, -1, 0, 0, 0, 0]
    dy_arr = [0, 0, 1, -1, 0, 0]
    dz_arr = [0, 0, 0, 0, 1, -1]
    bs = BLOCK_SIZE
    bs2 = bs*bs
    
    function has_block_local(bx, by, bz)
        if bx < 1 || bx > grid_dim_x || by < 1 || by > grid_dim_y || bz < 1 || bz > grid_dim_z
            return false
        end
        return block_ptr[bx, by, bz] > 0
    end
    
    while head <= length(q_block)
        b_curr = q_block[head]
        idx_curr = q_idx[head]
        head += 1
        
        rem = idx_curr - 1
        lz = rem ÷ bs2 + 1
        rem = rem % bs2
        ly = rem ÷ bs + 1
        lx = rem % bs + 1
        
        (bx, by, bz) = sorted_blocks[b_curr]
        
        for i in 1:6
            nlx = lx + dx_arr[i]
            nly = ly + dy_arr[i]
            nlz = lz + dz_arr[i]
            
            if nlx >= 1 && nlx <= bs && nly >= 1 && nly <= bs && nlz >= 1 && nlz <= bs
                if !visited[nlx, nly, nlz, b_curr] && !obstacle_arr[nlx, nly, nlz, b_curr]
                    visited[nlx, nly, nlz, b_curr] = true
                    push!(q_block, b_curr)
                    push!(q_idx, nlx + (nly-1)*bs + (nlz-1)*bs2)
                end
            else
                nbx = bx + dx_arr[i]
                nby = by + dy_arr[i]
                nbz = bz + dz_arr[i]
                
                if has_block_local(nbx, nby, nbz)
                    nb_idx = block_ptr[nbx, nby, nbz]
                    wlx = (nlx < 1) ? bs : (nlx > bs ? 1 : nlx)
                    wly = (nly < 1) ? bs : (nly > bs ? 1 : nly)
                    wlz = (nlz < 1) ? bs : (nlz > bs ? 1 : nlz)
                    
                    if !visited[wlx, wly, wlz, nb_idx] && !obstacle_arr[wlx, wly, wlz, nb_idx]
                        visited[wlx, wly, wlz, nb_idx] = true
                        push!(q_block, nb_idx)
                        push!(q_idx, wlx + (wly-1)*bs + (wlz-1)*bs2)
                    end
                end
            end
        end
    end
    
    filled_count = 0
    for i in 1:length(obstacle_arr)
        if !obstacle_arr[i] && !visited[i]
            obstacle_arr[i] = true
            filled_count += 1
        end
    end
    println("[Domain] Filled $filled_count interior voxels.")
end

@inline function smooth_sponge_profile(x::Float64, thickness::Float64)
    if x <= 0.0
        return 1.0
    elseif x >= thickness
        return 0.0
    else
        return 0.5 * (1.0 + cos(π * x / thickness))
    end
end

function apply_sponge!(sponge_arr, sorted_blocks, params, lvl_scale::Int)
    println("[Domain] Applying Sponge Layers...")
    dx = params.dx_coarse / lvl_scale
    
    Lx = params.domain_size[1]
    Ly = params.domain_size[2]
    Lz = params.domain_size[3]
    
    outlet_thickness = Lx * Float64(SPONGE_THICKNESS)
    inlet_thickness = Lx * 0.02  
    y_sponge_thickness = Ly * Float64(SPONGE_THICKNESS) * 0.5
    z_sponge_thickness = Lz * Float64(SPONGE_THICKNESS) * 0.5
    
    outlet_start = Lx - outlet_thickness
    y_top_start = Ly - y_sponge_thickness
    z_back_start = Lz - z_sponge_thickness
    
    outlet_strength = 1.0   
    inlet_strength = 0.3    
    wall_strength = 0.4       
    
    for i in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[i]
        
        for lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
            px = ((bx - 1) * BLOCK_SIZE + lx - 0.5) * dx
            py = ((by - 1) * BLOCK_SIZE + ly - 0.5) * dx
            pz = ((bz - 1) * BLOCK_SIZE + lz - 0.5) * dx
            
            sponge_val = 0.0
            
            if px > outlet_start
                dist_from_boundary = px - outlet_start
                s = smooth_sponge_profile(outlet_thickness - dist_from_boundary, outlet_thickness)
                sponge_val = max(sponge_val, s * outlet_strength)
            end
            
            if px < inlet_thickness
                s = smooth_sponge_profile(px, inlet_thickness)
                sponge_val = max(sponge_val, s * inlet_strength)
            end
            
            if !SYMMETRIC_ANALYSIS && py < y_sponge_thickness
                s = smooth_sponge_profile(py, y_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            if py > y_top_start
                dist_from_boundary = py - y_top_start
                s = smooth_sponge_profile(y_sponge_thickness - dist_from_boundary, y_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            if pz < z_sponge_thickness
                s = smooth_sponge_profile(pz, z_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            if pz > z_back_start
                dist_from_boundary = pz - z_back_start
                s = smooth_sponge_profile(z_sponge_thickness - dist_from_boundary, z_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            sponge_arr[lx, ly, lz, i] = Float32(sponge_val)
        end
    end
    
    sponge_cells = count(sponge_arr .> 0.0f0)
    total_cells = length(sponge_arr)
    max_sponge = maximum(sponge_arr)
    @printf("[Domain] Sponge: %.1f%% cells affected, max strength = %.3f\n", 
            100.0 * sponge_cells / total_cells, max_sponge)
end

"""
    compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, mesh, dx, mesh_offset)

Compute wall distances for cells adjacent to obstacles.
This is a simplified version that computes distance to nearest obstacle cell.
For more accurate results, use the geometry-based SDF computation.
"""
function compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, 
                                  mesh::Geometry.SolverMesh, dx::Float64, 
                                  mesh_offset::SVector{3,Float64})
    println("[Domain] Computing wall distances...")
    bs = BLOCK_SIZE
    n_blocks = length(sorted_blocks)
    
    # Build block lookup
    block_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        block_lookup[coord] = i
    end
    
    near_wall_count = 0
    
    @threads for b_idx in 1:n_blocks
        (bx, by, bz) = sorted_blocks[b_idx]
        
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            if obstacle_arr[lx, ly, lz, b_idx]
                continue  # Skip obstacle cells
            end
            
            # Check if any neighbor is an obstacle
            is_near_wall = false
            min_dist = 100.0f0
            
            for dz in -1:1, dy in -1:1, dx_off in -1:1
                if dx_off == 0 && dy == 0 && dz == 0
                    continue
                end
                
                nx, ny, nz = lx + dx_off, ly + dy, lz + dz
                nb_idx = b_idx
                
                # Handle block boundaries
                if nx < 1 || nx > bs || ny < 1 || ny > bs || nz < 1 || nz > bs
                    nbx = bx + (nx < 1 ? -1 : (nx > bs ? 1 : 0))
                    nby = by + (ny < 1 ? -1 : (ny > bs ? 1 : 0))
                    nbz = bz + (nz < 1 ? -1 : (nz > bs ? 1 : 0))
                    
                    if haskey(block_lookup, (nbx, nby, nbz))
                        nb_idx = block_lookup[(nbx, nby, nbz)]
                        nx = nx < 1 ? nx + bs : (nx > bs ? nx - bs : nx)
                        ny = ny < 1 ? ny + bs : (ny > bs ? ny - bs : ny)
                        nz = nz < 1 ? nz + bs : (nz > bs ? nz - bs : nz)
                    else
                        continue
                    end
                end
                
                if obstacle_arr[nx, ny, nz, nb_idx]
                    is_near_wall = true
                    dist = sqrt(Float32(dx_off^2 + dy^2 + dz^2)) * Float32(dx)
                    min_dist = min(min_dist, dist)
                end
            end
            
            if is_near_wall
                wall_dist_arr[lx, ly, lz, b_idx] = min_dist
                near_wall_count += 1
            end
        end
    end
    
    println("[Domain] Found $near_wall_count near-wall cells")
end

function setup_multilevel_domain(mesh::Geometry.SolverMesh, params)
    println("\n" * "="^70)
    println("[Setup] Multi-Level Domain Generation")
    if REFINEMENT_STRATEGY == :geometry_first
        println("        Strategy: Geometry-First (Direct Mesh Check)")
    else
        println("        Strategy: Topology-Legacy (Inherited Obstacles)")
    end
    if BOUNDARY_METHOD == :bouzidi
        println("        Boundary: Bouzidi IBM (finest $BOUZIDI_LEVELS level(s))")
    else
        println("        Boundary: Simple Bounce-Back")
    end
    println("="^70)
    
    num_levels = params.num_levels
    mesh_offset = SVector(params.mesh_offset[1], params.mesh_offset[2], params.mesh_offset[3])
    
    placed_mesh_min = SVector(params.mesh_min...) + mesh_offset
    placed_mesh_max = SVector(params.mesh_max...) + mesh_offset
    
    wake_start_x = placed_mesh_max[1] - (params.reference_length * 0.1)
    wake_end_x = placed_mesh_max[1] + (params.reference_length * WAKE_REFINEMENT_LENGTH)
    
    wake_center_y = (placed_mesh_min[2] + placed_mesh_max[2]) / 2.0
    wake_center_z = (placed_mesh_min[3] + placed_mesh_max[3]) / 2.0
    
    wake_width = (placed_mesh_max[2] - placed_mesh_min[2]) * WAKE_REFINEMENT_WIDTH_FACTOR
    wake_height = (placed_mesh_max[3] - placed_mesh_min[3]) * WAKE_REFINEMENT_HEIGHT_FACTOR
    
    wake_min_y = wake_center_y - wake_width/2.0
    wake_max_y = wake_center_y + wake_width/2.0
    wake_min_z = wake_center_z - wake_height/2.0
    wake_max_z = wake_center_z + wake_height/2.0
    
    grids = Vector{BlockLevel}()
    
    for lvl in 1:num_levels
        println("\n--- Level $lvl ---")
        
        scale = 2^(lvl - 1)
        dx = params.dx_coarse / scale
        dt = 1.0f0 / Float32(scale)
        tau = params.tau_levels[lvl]
        
        bx_max = params.bx_max * scale
        by_max = params.by_max * scale
        bz_max = params.bz_max * scale
        
        active_set = Set{Tuple{Int,Int,Int}}()
        
        if lvl == 1
            println("  Generating Full Wind Tunnel for Level 1...")
            for bz in 1:bz_max, by in 1:by_max, bx in 1:bx_max
                push!(active_set, (bx, by, bz))
            end
            println("  Full domain: $(bx_max)×$(by_max)×$(bz_max) = $(length(active_set)) blocks")
        else
            prev_level = grids[lvl-1]
            prev_scale = 2^(lvl - 2)
            prev_dx = params.dx_coarse / prev_scale
            prev_bs_phys = BLOCK_SIZE * prev_dx
            
            if REFINEMENT_STRATEGY == :geometry_first
                surface_blocks = get_active_blocks_for_level(mesh, dx, mesh_offset, bx_max, by_max, bz_max)
                union!(active_set, surface_blocks)
                
                if ENABLE_WAKE_REFINEMENT
                    for (cbx, cby, cbz) in prev_level.active_block_coords
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys
                        
                        overlap_x = (b_min_x <= wake_end_x) && (b_max_x >= wake_start_x)
                        overlap_y = (b_min_y <= wake_max_y) && (b_max_y >= wake_min_y)
                        overlap_z = (b_min_z <= wake_max_z) && (b_max_z >= wake_min_z)
                        
                        if overlap_x && overlap_y && overlap_z
                            for dbz in 0:1, dby in 0:1, dbx in 0:1
                                fbx = 2*cbx - 1 + dbx
                                fby = 2*cby - 1 + dby
                                fbz = 2*cbz - 1 + dbz
                                if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                    push!(active_set, (fbx, fby, fbz))
                                end
                            end
                        end
                    end
                end
                
                prev_level_set = Set(grids[lvl-1].active_block_coords)
                orphans = 0
                final_set = Set{Tuple{Int,Int,Int}}()
                for (bx, by, bz) in active_set
                    pbx = (bx + 1) ÷ 2
                    pby = (by + 1) ÷ 2
                    pbz = (bz + 1) ÷ 2
                    if (pbx, pby, pbz) in prev_level_set
                        push!(final_set, (bx, by, bz))
                    else
                        orphans += 1
                    end
                end
                active_set = final_set
            else
                for (b_idx, (cbx, cby, cbz)) in enumerate(prev_level.active_block_coords)
                    is_surface = any(view(prev_level.obstacle, :, :, :, b_idx))
                    
                    is_wake = false
                    if ENABLE_WAKE_REFINEMENT && !is_surface
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys
                        
                        overlap_x = (b_min_x <= wake_end_x) && (b_max_x >= wake_start_x)
                        overlap_y = (b_min_y <= wake_max_y) && (b_max_y >= wake_min_y)
                        overlap_z = (b_min_z <= wake_max_z) && (b_max_z >= wake_min_z)
                        is_wake = overlap_x && overlap_y && overlap_z
                    end
                    
                    if is_surface || is_wake
                        for dbz in 0:1, dby in 0:1, dbx in 0:1
                            fbx = 2*cbx - 1 + dbx
                            fby = 2*cby - 1 + dby
                            fbz = 2*cbz - 1 + dbz
                            if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                push!(active_set, (fbx, fby, fbz))
                            end
                        end
                    end
                end
            end
        end
        
        n_before_halo = length(active_set)
        
        add_halo_blocks_with_siblings!(active_set, REFINEMENT_MARGIN, bx_max, by_max, bz_max)
        ensure_complete_parent_coverage!(active_set, bx_max, by_max, bz_max)
        
        n_after_halo = length(active_set)
        if lvl > 1
            println("  Added $(n_after_halo - n_before_halo) halo blocks")
        end
        
        sorted_blocks = sort(collect(active_set))
        n_blocks = length(sorted_blocks)
        
        nb_table = build_neighbor_table(sorted_blocks, bx_max, by_max, bz_max)
        
        bs = BLOCK_SIZE
        obstacle_arr = falses(bs, bs, bs, n_blocks)
        sponge_arr = zeros(Float32, bs, bs, bs, n_blocks)
        wall_dist_arr = fill(100.0f0, bs, bs, bs, n_blocks)
        
        block_ptr = zeros(Int32, bx_max, by_max, bz_max)
        for (idx, (bx, by, bz)) in enumerate(sorted_blocks)
            block_ptr[bx, by, bz] = Int32(idx)
        end
        
        voxelize_blocks!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)
        perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, bx_max, by_max, bz_max)
        
        apply_sponge!(sponge_arr, sorted_blocks, params, scale)
        
        # Compute wall distances for wall model
        if WALL_MODEL_ENABLED
            compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, mesh, dx, mesh_offset)
        end
        
        # Check if this level should use Bouzidi
        use_bouzidi = should_use_bouzidi(lvl, num_levels, BOUNDARY_METHOD, BOUZIDI_LEVELS)
        
        # Initialize Bouzidi data
        bouzidi_q_map = nothing
        bouzidi_cell_block = nothing
        bouzidi_cell_x = nothing
        bouzidi_cell_y = nothing
        bouzidi_cell_z = nothing
        n_boundary_cells = 0
        
        if use_bouzidi
            println("[Bouzidi] Computing Q-map for Level $lvl...")
            
            # FIXED: Use compute_bouzidi_qmap_sparse directly (returns values instead of mutating)
            q_map_cpu, cell_block, cell_x, cell_y, cell_z, n_b = 
                compute_bouzidi_qmap_sparse(sorted_blocks, mesh, dx, mesh_offset, BLOCK_SIZE)
            
            bouzidi_q_map = q_map_cpu
            bouzidi_cell_block = cell_block
            bouzidi_cell_x = cell_x
            bouzidi_cell_y = cell_y
            bouzidi_cell_z = cell_z
            n_boundary_cells = n_b
            
            println("[Bouzidi] Level $lvl: $n_boundary_cells boundary cells")
        end
        
        level = BlockLevel(lvl, sorted_blocks, nb_table, Float32(dx), dt, tau;
                          bouzidi_q_map = bouzidi_q_map,
                          bouzidi_cell_block = bouzidi_cell_block,
                          bouzidi_cell_x = bouzidi_cell_x,
                          bouzidi_cell_y = bouzidi_cell_y,
                          bouzidi_cell_z = bouzidi_cell_z,
                          n_boundary_cells = n_boundary_cells)
        
        level.obstacle .= obstacle_arr
        level.sponge .= sponge_arr
        level.wall_dist .= wall_dist_arr
        
        push!(grids, level)
        
        n_voxels = n_blocks * BLOCK_SIZE^3
        @printf("  Total: %d blocks, %.2f M voxels\n", n_blocks, n_voxels/1e6)
        if use_bouzidi
            println("  Boundary: Bouzidi IBM ($n_boundary_cells cells, sparse)")
        else
            println("  Boundary: Simple bounce-back")
        end
    end
    
    println("\n[Verify] Checking parent coverage...")
    for lvl in 2:num_levels
        fine_set = Set(grids[lvl].active_block_coords)
        coarse_set = Set(grids[lvl-1].active_block_coords)
        missing = 0
        for (fbx, fby, fbz) in fine_set
            pbx = (fbx + 1) ÷ 2
            pby = (fby + 1) ÷ 2
            pbz = (fbz + 1) ÷ 2
            if !((pbx, pby, pbz) in coarse_set)
                missing += 1
            end
        end
        println("  Level $lvl: $(missing == 0 ? "✓" : "⚠") (missing parents: $missing)")
    end
    
    println("\n" * "="^70)
    total_blocks = sum(length(g.active_block_coords) for g in grids)
    total_cells = sum(length(g.active_block_coords) * BLOCK_SIZE^3 for g in grids)
    total_bouzidi_mem = sum(g.bouzidi_enabled ? sizeof(g.bouzidi_q_map)/1024^2 : 0 for g in grids)
    total_sparse_mem = sum(g.bouzidi_enabled ? 
        (sizeof(g.bouzidi_cell_block) + sizeof(g.bouzidi_cell_x) + 
         sizeof(g.bouzidi_cell_y) + sizeof(g.bouzidi_cell_z))/1024 : 0 for g in grids)
    
    @printf("[Done] %d levels, %d blocks, %.2f M cells\n", num_levels, total_blocks, total_cells/1e6)
    if BOUNDARY_METHOD == :bouzidi
        @printf("[Done] Bouzidi: %.1f MB q_map + %.1f KB coords\n", total_bouzidi_mem, total_sparse_mem)
    end
    println("="^70)
    
    return grids
end

function setup_multilevel_domain(stl_path::String; num_levels=NUM_LEVELS)
    println("[Domain] Loading: $stl_path")
    if !isfile(stl_path); error("STL file not found: $stl_path"); end
    
    mesh = Geometry.load_mesh(stl_path, scale=Float64(STL_SCALE))
    bounds = Geometry.compute_mesh_bounds(mesh)
    
    compute_domain_from_mesh(Tuple(bounds.min_bounds), Tuple(bounds.max_bounds))
    params = get_domain_params()
    print_domain_summary()
    
    return setup_multilevel_domain(mesh, params)
end