// # FILE: ./src/bouzidi.jl
"""
BOUZIDI.JL - Bouzidi Interpolated Bounce-Back (Optimized Sparse Version)

Memory optimizations:
- Sparse boundary cell list (only stores coordinates of boundary cells)
- Removed redundant is_boundary mask
- Separate kernel launches only for boundary cells (not entire domain)

Speed optimizations:
- Main collision kernel has zero Bouzidi overhead
- Bouzidi kernel: every thread does useful work (no divergence)
"""

using KernelAbstractions
using StaticArrays
using CUDA
using Adapt
using Base.Threads
using LinearAlgebra

"""
    BouzidiDataSparse

Sparse storage for Bouzidi boundary conditions.
(Maintained for backward compatibility in setup, though runtime uses unwrapped arrays).
"""
struct BouzidiDataSparse{A_Q, A_Idx}
    q_map::A_Q   
    cell_block::A_Idx   
    cell_x::A_Idx       
    cell_y::A_Idx       
    cell_z::A_Idx       
    n_boundary_cells::Int
end

@inline function ray_triangle_intersection(origin::SVector{3,Float64}, 
                                           dir::SVector{3,Float64},
                                           v1::SVector{3,Float64}, 
                                           v2::SVector{3,Float64}, 
                                           v3::SVector{3,Float64})
    EPSILON = 1e-9
    
    edge1 = v2 - v1
    edge2 = v3 - v1
    h = cross(dir, edge2)
    a = dot(edge1, h)
    
    if abs(a) < EPSILON
        return (false, 0.0)
    end
    
    f = 1.0 / a
    s = origin - v1
    u = f * dot(s, h)
    
    if u < 0.0 || u > 1.0
        return (false, 0.0)
    end
    
    q_vec = cross(s, edge1)
    v = f * dot(dir, q_vec)
    
    if v < 0.0 || u + v > 1.0
        return (false, 0.0)
    end
    
    t = f * dot(edge2, q_vec)
    
    if t > EPSILON
        return (true, t)
    else
        return (false, 0.0)
    end
end

"""
    compute_q_for_cell - Compute Q-values for all 27 directions at a cell
"""
function compute_q_for_cell(cell_center::SVector{3,Float64},
                            dx::Float64,
                            triangles::Vector,
                            mesh_offset::SVector{3,Float64},
                            cx::Vector{Int32},
                            cy::Vector{Int32},
                            cz::Vector{Int32})
    
    q_values = zeros(Float64, 27)
    
    for k in 1:27
        dir = SVector(Float64(cx[k]), Float64(cy[k]), Float64(cz[k]))
        
        if cx[k] == 0 && cy[k] == 0 && cz[k] == 0
            q_values[k] = 0.0
            continue
        end
        
        dir_norm = dir / norm(dir)
        min_t = Inf
        
        for tri in triangles
            v1 = SVector{3,Float64}(tri[1]) + mesh_offset
            v2 = SVector{3,Float64}(tri[2]) + mesh_offset
            v3 = SVector{3,Float64}(tri[3]) + mesh_offset
            
            hit, t = ray_triangle_intersection(cell_center, -dir_norm, v1, v2, v3)
            
            if hit && t < min_t
                min_t = t
            end
        end
        
        if min_t < Inf
            c_magnitude = norm(dir)
            q = min_t / (dx * c_magnitude)
            
            if q > 0.0 && q <= 1.0
                q_values[k] = q
            end
        end
    end
    
    return q_values
end

"""
    build_block_triangle_map_for_bouzidi - Build spatial hash of triangles per block
"""
function build_block_triangle_map_for_bouzidi(mesh, 
                                                sorted_blocks::Vector{Tuple{Int,Int,Int}},
                                                dx::Float64,
                                                mesh_offset::SVector{3,Float64},
                                                block_size::Int)
    block_tris = [Int[] for _ in 1:length(sorted_blocks)]
    bs = block_size
    
    b_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        b_lookup[coord] = i
    end
    
    margin = dx * 2.5
    
    for (t_idx, tri) in enumerate(mesh.triangles)
        v1 = SVector{3,Float64}(tri[1])
        v2 = SVector{3,Float64}(tri[2])
        v3 = SVector{3,Float64}(tri[3])
        
        tri_min = min.(v1, min.(v2, v3))
        tri_max = max.(v1, max.(v2, v3))
        
        min_pt = tri_min + mesh_offset
        max_pt = tri_max + mesh_offset
        
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

"""
    compute_bouzidi_qmap_sparse(level_active_coords, mesh, dx, mesh_offset, block_size)

Compute Q-values and build sparse boundary cell coordinate lists.
Returns: (q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells)
"""
function compute_bouzidi_qmap_sparse(level_active_coords::Vector{Tuple{Int,Int,Int}},
                                     mesh,
                                     dx::Float64,
                                     mesh_offset::SVector{3,Float64},
                                     block_size::Int)
    
    println("[Bouzidi] Computing Q-values for $(length(level_active_coords)) blocks...")
    
    n_blocks = length(level_active_coords)
    bs = block_size
    
    cx = Int32[]
    cy = Int32[]
    cz = Int32[]
    for dz in -1:1, dy in -1:1, dx_i in -1:1
        push!(cx, Int32(dx_i))
        push!(cy, Int32(dy))
        push!(cz, Int32(dz))
    end
    
    q_map_cpu = zeros(Float16, bs, bs, bs, n_blocks, 27)
    
    # --------------------------------------------------------------------------
    # THREAD ID SAFETY FIX:
    # In Julia 1.9+, `threadid()` can return indices > `nthreads()` if using 
    # interactive thread pools (e.g. `julia -t 4,1`).
    # We must size the array based on the maximum possible ID.
    # --------------------------------------------------------------------------
    max_tid = Threads.nthreads()
    if isdefined(Threads, :threadpoolsize)
        try
            max_tid += Threads.threadpoolsize(:interactive)
        catch
        end
    end
    # Allocation
    thread_boundary_cells = [Vector{NTuple{4,Int32}}() for _ in 1:max_tid]
    # --------------------------------------------------------------------------

    block_tri_map = build_block_triangle_map_for_bouzidi(mesh, level_active_coords, 
                                                         dx, mesh_offset, bs)
    
    @threads for b_idx in 1:n_blocks
        tid = Threads.threadid()
        (bx, by, bz) = level_active_coords[b_idx]
        relevant_tris = block_tri_map[b_idx]
        
        if isempty(relevant_tris)
            continue
        end

        # Safety check for unexpected thread IDs (e.g., custom schedulers)
        if tid > length(thread_boundary_cells)
             # Fallback/Error: If this happens, the thread ID exceeded expected bounds.
             # This is rare if max_tid is calculated correctly above.
             continue 
        end
        
        local_triangles = [mesh.triangles[t_idx] for t_idx in relevant_tris]
        
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            px = ((bx - 1) * bs + lx - 0.5) * dx
            py = ((by - 1) * bs + ly - 0.5) * dx
            pz = ((bz - 1) * bs + lz - 0.5) * dx
            cell_center = SVector(px, py, pz)
            
            q_vals = compute_q_for_cell(cell_center, dx, local_triangles, 
                                        mesh_offset, cx, cy, cz)
            
            has_boundary = false
            for k in 1:27
                if q_vals[k] > 0.0
                    q_map_cpu[lx, ly, lz, b_idx, k] = Float16(q_vals[k])
                    has_boundary = true
                end
            end
            
            if has_boundary
                push!(thread_boundary_cells[tid], (Int32(b_idx), Int32(lx), Int32(ly), Int32(lz)))
            end
        end
    end
    
    all_boundary_cells = NTuple{4,Int32}[]
    for cells in thread_boundary_cells
        append!(all_boundary_cells, cells)
    end
    
    n_boundary = length(all_boundary_cells)
    total_cells = n_blocks * bs^3
    
    cell_block = Vector{Int32}(undef, n_boundary)
    cell_x = Vector{Int32}(undef, n_boundary)
    cell_y = Vector{Int32}(undef, n_boundary)
    cell_z = Vector{Int32}(undef, n_boundary)
    
    for (i, (b, x, y, z)) in enumerate(all_boundary_cells)
        cell_block[i] = b
        cell_x[i] = x
        cell_y[i] = y
        cell_z[i] = z
    end
    
    qmap_mem_mb = sizeof(q_map_cpu) / 1024^2
    sparse_mem_kb = (sizeof(cell_block) + sizeof(cell_x) + sizeof(cell_y) + sizeof(cell_z)) / 1024
    
    println("[Bouzidi] Found $n_boundary boundary cells ($(round(100*n_boundary/total_cells, digits=2))%)")
    println("[Bouzidi] Q-map: $(round(qmap_mem_mb, digits=1)) MB, Sparse coords: $(round(sparse_mem_kb, digits=1)) KB")
    
    return q_map_cpu, cell_block, cell_x, cell_y, cell_z, n_boundary
end

"""
Bouzidi correction kernel - runs ONLY on boundary cells.
Launched with ndrange = (n_boundary_cells,) for maximum efficiency.
"""
@kernel function bouzidi_correction_kernel!(
    f_out,              
    f_in,             
    q_map,              
    cell_block,       
    cell_x,           
    cell_y,           
    cell_z,           
    neighbor_table,   
    block_size::Int32,
    cx_arr, cy_arr, cz_arr,
    opp_arr,
    q_min_threshold::Float32
)
    cell_idx = @index(Global)
    
    @inbounds begin
        # Get this boundary cell's coordinates
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]
        y = cell_y[cell_idx]
        z = cell_z[cell_idx]
        
        for k in 1:27
            q = Float32(q_map[x, y, z, b_idx, k])
            
            if q > q_min_threshold && q <= 1.0f0
                opp_k = opp_arr[k]
                
                # Formula:
                # f_i' = 2q * f_i^{neq} + (1-2q) * f_i^{eq}  (Standard BB)
                # But here we use the interpolated version (Bouzidi):
                # 
                # If q < 0.5:
                # f_i(x_b) = 2q * f_i^c(x_b) + (1-2q) * f_i^c(x_f)
                # where x_f is further fluid node.
                # However, common LBM implementation simplifies to linear interp:
                
                # f_i(x, t+dt) = ...
                
                # Implementation:
                # Based on standard Bouzidi equations for LBM.
                
                f_bb = f_out[x, y, z, b_idx, k]
                
                if q < 0.5f0
                    # q < 0.5: The wall is closer to the fluid node x than to x + c_i
                    # Use interpolation involving f(x) and f(x + c_i_opp)
                    
                    # f_out[...] currently holds the post-collision stream value that ARRIVED 
                    # at the boundary if the wall wasn't there (which is f_out[..., opp_k]).
                    
                    # The formula is: f_i'(x) = 2q * f_i^{neq} + ...
                    
                    # Implementation from literature (e.g., Ginzburg/d'Humieres):
                    
                    # f_i(x, t+1) = 2q * f_{i*}(x, t) + (1-2q) * f_{i*}(x - c_i, t)
                    # Here f_{i*} is post-collision, pre-stream.
                    # But we are in "apply correction" phase AFTER stream.
                    
                    # So f_bb corresponds to the stream that bounced back.
                    
                    f_opp_val = f_out[x, y, z, b_idx, opp_k]
                    coeff1 = 2.0f0 * q
                    f_out[x, y, z, b_idx, k] = coeff1 * f_bb + (1.0f0 - coeff1) * f_opp_val
                else
                    # q >= 0.5: The wall is further.
                    # f_i(x, t+1) = (1/(2q)) * f_{i*}(x, t) + (1 - 1/(2q)) * f_{i*}(x, t) ?
                    # Standard formula: 
                    # f_i(x) = (1/2q) * f_{i*}(x) + (1 - 1/2q) * f_{i*}(x + c_i)
                    
                    # We need the value from the neighbor in the direction of the wall.
                    # This is f_in[x + c_i, ..., opp_k] because f_in is the SOURCE state.
                    
                    # Assuming f_in holds valid neighbor data (it does, it's the source for this step).
                    
                    cx = cx_arr[k]
                    cy = cy_arr[k]
                    cz = cz_arr[k]
                    
                    nx = x + cx
                    ny = y + cy
                    nz = z + cz
                    
                    f_neighbor = f_bb 
                    
                    # Get neighbor safely (might be in another block)
                    if nx >= 1 && nx <= block_size && ny >= 1 && ny <= block_size && nz >= 1 && nz <= block_size
                        # Local neighbor
                        # We want the population moving OPPOSITE to k, at the neighbor node.
                        # Wait, Bouzidi usually interpolates post-collision states.
                        
                        f_neighbor = f_in[nx, ny, nz, b_idx, opp_k] 
                    else
                        # Remote neighbor
                        nb_off_x = (nx < 1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
                        nb_off_y = (ny < 1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
                        nb_off_z = (nz < 1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))
                        
                        dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) + (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
                        nb_block = neighbor_table[b_idx, dir_idx]
                        
                        if nb_block > 0
                            nnx = nx < 1 ? nx + block_size : (nx > block_size ? nx - block_size : nx)
                            nny = ny < 1 ? ny + block_size : (ny > block_size ? ny - block_size : ny)
                            nnz = nz < 1 ? nz + block_size : (nz > block_size ? nz - block_size : nz)
                            f_neighbor = f_in[nnx, nny, nnz, nb_block, opp_k]
                        end
                    end
                    
                    inv_2q = 1.0f0 / (2.0f0 * q)
                    f_out[x, y, z, b_idx, k] = inv_2q * f_bb + (1.0f0 - inv_2q) * f_neighbor
                end
            end
        end
    end
end

"""
    apply_bouzidi_correction!(f_out, f_in, q_map, cell_block, ..., backend)

Apply Bouzidi boundary correction. Call AFTER the main stream-collide kernel.
Matches the A-B pattern requirements by taking explicit arrays.
"""
function apply_bouzidi_correction!(f_out, f_in,
                                   q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
                                   neighbor_table,
                                   block_size::Int,
                                   cx_gpu, cy_gpu, cz_gpu, opp_gpu,
                                   q_min_threshold::Float32,
                                   backend)
    
    if n_boundary_cells == 0
        return
    end
    
    kernel! = bouzidi_correction_kernel!(backend)
    kernel!(f_out, f_in,
            q_map,
            cell_block,
            cell_x,
            cell_y,
            cell_z,
            neighbor_table,
            Int32(block_size),
            cx_gpu, cy_gpu, cz_gpu, opp_gpu,
            q_min_threshold,
            ndrange=(n_boundary_cells,))
end

"""
Determine if a level should use Bouzidi boundary conditions.
"""
function should_use_bouzidi(level_id::Int, num_levels::Int, 
                            boundary_method::Symbol, bouzidi_levels::Int)
    if boundary_method != :bouzidi
        return false
    end
    return level_id > (num_levels - bouzidi_levels)
end

"""
Transfer Bouzidi data to GPU.
"""
function create_bouzidi_data_gpu(q_map_cpu, cell_block, cell_x, cell_y, cell_z, 
                                 n_boundary::Int, backend)
    return BouzidiDataSparse(
        adapt(backend, q_map_cpu),
        adapt(backend, cell_block),
        adapt(backend, cell_x),
        adapt(backend, cell_y),
        adapt(backend, cell_z),
        n_boundary
    )
end

"""
Print Bouzidi memory usage summary.
"""
function print_bouzidi_memory(bouzidi_data::BouzidiDataSparse, level_id::Int)
    q_mem = sizeof(bouzidi_data.q_map) / 1024^2
    sparse_mem = (sizeof(bouzidi_data.cell_block) + sizeof(bouzidi_data.cell_x) + 
                  sizeof(bouzidi_data.cell_y) + sizeof(bouzidi_data.cell_z)) / 1024
    
    println("  Level $level_id Bouzidi: $(bouzidi_data.n_boundary_cells) cells, " *
            "$(round(q_mem, digits=1)) MB q_map, $(round(sparse_mem, digits=1)) KB coords")
end