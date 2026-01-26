// # FILE: .\src\geometry.jl";

module Geometry

using LinearAlgebra
using Printf
using StaticArrays

# Export the struct and functions used by domain.jl
export load_mesh, compute_mesh_bounds
export mark_domain_sat!, compute_q_values!, compute_narrow_band_sdf!
export SolverMesh 

"""
    SolverMesh (Geometry)

Holds the triangle soup and bounding box of the input geometry.
Stored as Tuples for lightweight loading, converted to SVectors for processing.
"""
struct SolverMesh
    triangles::Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}
    min_bounds::Tuple{Float64, Float64, Float64}
    max_bounds::Tuple{Float64, Float64, Float64}
end

# -------------------------------------------------------------------------
# STL LOADER (Standalone - No external dependencies)
# -------------------------------------------------------------------------

function parse_binary_stl(io::IO, scale::Float64)
    skip(io, 80) # Skip header
    count = read(io, UInt32)
    
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}(undef, count)
    
    for i in 1:count
        skip(io, 12) # Normal
        
        x1 = Float64(read(io, Float32)) * scale
        y1 = Float64(read(io, Float32)) * scale
        z1 = Float64(read(io, Float32)) * scale
        
        x2 = Float64(read(io, Float32)) * scale
        y2 = Float64(read(io, Float32)) * scale
        z2 = Float64(read(io, Float32)) * scale
        
        x3 = Float64(read(io, Float32)) * scale
        y3 = Float64(read(io, Float32)) * scale
        z3 = Float64(read(io, Float32)) * scale
        
        skip(io, 2) # Attribute byte count
        
        triangles[i] = ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3))
    end
    return triangles
end

function parse_ascii_stl(filename::String, scale::Float64)
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}()
    current_tri = Vector{Tuple{Float64,Float64,Float64}}()
    
    for line in eachline(filename)
        s = strip(line)
        if startswith(s, "vertex")
            parts = split(s)
            if length(parts) >= 4
                x = parse(Float64, parts[2]) * scale
                y = parse(Float64, parts[3]) * scale
                z = parse(Float64, parts[4]) * scale
                push!(current_tri, (x, y, z))
            end
        elseif startswith(s, "endloop")
            if length(current_tri) == 3
                push!(triangles, (current_tri[1], current_tri[2], current_tri[3]))
            end
            empty!(current_tri)
        end
    end
    return triangles
end

function load_mesh(filename::String; scale=1.0)
    if !isfile(filename); error("STL file not found: $filename"); end
    println("[Geometry] Loading STL from: $filename")
    
    triangles = nothing
    is_binary = true
    
    open(filename, "r") do io
        if filesize(filename) < 84
            is_binary = false
        else
            header = String(read(io, 5))
            if startswith(lowercase(header), "solid")
                seek(io, 80)
                count = read(io, UInt32)
                if filesize(filename) != 84 + count * 50
                    is_binary = false
                end
            end
        end
    end
    
    if is_binary
        open(filename, "r") do io; triangles = parse_binary_stl(io, scale); end
        println("[Geometry] Format: Binary STL")
    else
        triangles = parse_ascii_stl(filename, scale)
        println("[Geometry] Format: ASCII STL")
    end

    if isempty(triangles); error("No triangles loaded."); end
    
    min_x, min_y, min_z = Inf, Inf, Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf

    for tri in triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end
    
    println("[Geometry] Loaded $(length(triangles)) triangles.")
    println("[Geometry] Bounds: [$(round(min_x,digits=3)), $(round(max_x,digits=3))] x [$(round(min_y,digits=3)), $(round(max_y,digits=3))] x [$(round(min_z,digits=3)), $(round(max_z,digits=3))]")

    return SolverMesh(triangles, (min_x, min_y, min_z), (max_x, max_y, max_z))
end

function compute_mesh_bounds(mesh::SolverMesh)
    return (min_bounds=mesh.min_bounds, max_bounds=mesh.max_bounds)
end

# -------------------------------------------------------------------------
# GEOMETRY PROCESSING KERNELS (With SVector Conversion)
# -------------------------------------------------------------------------

@inline function project_triangle(v1, v2, v3, axis)
    d1 = dot(v1, axis)
    d2 = dot(v2, axis)
    d3 = dot(v3, axis)
    return min(d1, d2, d3), max(d1, d2, d3)
end

@inline function project_box(box_half_size, axis)
    return box_half_size * (abs(axis[1]) + abs(axis[2]) + abs(axis[3]))
end

@inline function sat_overlap(v1, v2, v3, box_center, box_half_size, axis)
    # Relative to box center
    min_tri, max_tri = project_triangle(v1 - box_center, v2 - box_center, v3 - box_center, axis)
    r_box = project_box(box_half_size, axis)
    return !((min_tri > r_box) || (max_tri < -r_box))
end

function test_triangle_box_overlap(center_tuple, half_size_val, v1_t, v2_t, v3_t)
    # Convert Tuples to SVector for math
    center = SVector(center_tuple)
    half_size = SVector(half_size_val, half_size_val, half_size_val) # Assuming cube voxels usually, but logic allows vector
    v1 = SVector(v1_t)
    v2 = SVector(v2_t)
    v3 = SVector(v3_t)

    # 1. Box normals
    if !sat_overlap(v1, v2, v3, center, half_size, SVector(1.0, 0.0, 0.0)); return false; end
    if !sat_overlap(v1, v2, v3, center, half_size, SVector(0.0, 1.0, 0.0)); return false; end
    if !sat_overlap(v1, v2, v3, center, half_size, SVector(0.0, 0.0, 1.0)); return false; end

    # 2. Triangle normal
    e1 = v2 - v1
    e2 = v3 - v1
    n = cross(e1, e2)
    if !sat_overlap(v1, v2, v3, center, half_size, n); return false; end

    # 3. Edge cross products
    box_axes = (SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))
    tri_edges = (e1, v3 - v2, v1 - v3)
    
    for i in 1:3
        for j in 1:3
            axis = cross(box_axes[i], tri_edges[j])
            if sum(abs, axis) > 1e-9
                if !sat_overlap(v1, v2, v3, center, half_size, axis); return false; end
            end
        end
    end
    return true
end

function intersect_ray_triangle(origin_t, dir_t, v1_t, v2_t, v3_t)
    # Convert to SVector
    origin = SVector(origin_t)
    dir    = SVector(dir_t)
    v1     = SVector(v1_t)
    v2     = SVector(v2_t)
    v3     = SVector(v3_t)

    epsilon = 1e-7
    edge1 = v2 - v1
    edge2 = v3 - v1
    h = cross(dir, edge2)
    a = dot(edge1, h)
    
    if a > -epsilon && a < epsilon; return Inf; end
    
    f = 1.0 / a
    s = origin - v1
    u = f * dot(s, h)
    
    if u < 0.0 || u > 1.0; return Inf; end
    
    q = cross(s, edge1)
    v = f * dot(dir, q)
    
    if v < 0.0 || u + v > 1.0; return Inf; end
    
    t = f * dot(edge2, q)
    
    if t > epsilon; return t; end
    return Inf
end

# -------------------------------------------------------------------------
# OPTIMIZED SPATIAL MAPPING
# -------------------------------------------------------------------------

"""
    build_triangle_block_map(mesh, active_coords, dx, mesh_offset, margin_factor)

Pre-bins triangles into blocks. 
Returns a vector of indices, where map[b_idx] = [t_idx1, t_idx2...]
"""
function build_triangle_block_map(mesh::SolverMesh, active_coords, dx, mesh_offset, margin_factor=2.0)
    n_blocks = length(active_coords)
    block_tris = [Int[] for _ in 1:n_blocks]
    
    # Create lookup for block index: (bx,by,bz) -> linear_index
    block_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, c) in enumerate(active_coords)
        block_lookup[c] = i
    end
    
    bs = 8 # Block size hardcoded or passed. Assuming 8 for now.
    phys_bs = bs * dx
    margin = margin_factor * dx # Margin in physical units
    
    # Iterate all triangles and add to relevant blocks
    for (t_idx, tri) in enumerate(mesh.triangles)
        # Bounding box of triangle + margin
        v1 = SVector(tri[1]) .+ mesh_offset
        v2 = SVector(tri[2]) .+ mesh_offset
        v3 = SVector(tri[3]) .+ mesh_offset
        
        min_x = min(v1[1], v2[1], v3[1]) - margin
        max_x = max(v1[1], v2[1], v3[1]) + margin
        min_y = min(v1[2], v2[2], v3[2]) - margin
        max_y = max(v1[2], v2[2], v3[2]) + margin
        min_z = min(v1[3], v2[3], v3[3]) - margin
        max_z = max(v1[3], v2[3], v3[3]) + margin
        
        # Convert to block coordinates
        min_bx = floor(Int, min_x / phys_bs) + 1
        max_bx = floor(Int, max_x / phys_bs) + 1
        min_by = floor(Int, min_y / phys_bs) + 1
        max_by = floor(Int, max_y / phys_bs) + 1
        min_bz = floor(Int, min_z / phys_bs) + 1
        max_bz = floor(Int, max_z / phys_bs) + 1
        
        for bz in min_bz:max_bz
            for by in min_by:max_by
                for bx in min_bx:max_bx
                    if haskey(block_lookup, (bx, by, bz))
                        idx = block_lookup[(bx, by, bz)]
                        push!(block_tris[idx], t_idx)
                    end
                end
            end
        end
    end
    
    return block_tris
end

# -------------------------------------------------------------------------
# HIGH-LEVEL PROCESSING
# -------------------------------------------------------------------------

function mark_domain_sat!(level, mesh::SolverMesh, mesh_offset_t)
    println("[Domain] SAT Surface Marking for Level $(level.level_id)...")
    
    obstacles = Array(level.obstacle) 
    nx, ny, nz = size(obstacles)[1:3]
    active_coords = level.active_block_coords
    n_blocks = length(active_coords)
    dx = level.dx
    half_size_val = dx / 2.0
    
    # 1. Build Spatial Map
    println("[Domain] Binning triangles for optimization...")
    tri_map = build_triangle_block_map(mesh, active_coords, dx, SVector(mesh_offset_t))
    
    count = 0
    mesh_offset = SVector(mesh_offset_t)
    
    # 2. Iterate Blocks
    Base.Threads.@threads for b in 1:n_blocks
        local_tris_idx = tri_map[b]
        if isempty(local_tris_idx)
            continue
        end
        
        bx, by, bz = active_coords[b]
        min_bx = (bx-1)*nx * dx
        min_by = (by-1)*ny * dx
        min_bz = (bz-1)*nz * dx
        
        for k in 1:nz
            for j in 1:ny
                for i in 1:nx
                    if !obstacles[i, j, k, b]
                        cx = min_bx + (i-0.5)*dx
                        cy = min_by + (j-0.5)*dx
                        cz = min_bz + (k-0.5)*dx
                        
                        center_t = (cx, cy, cz)
                        
                        # Only check triangles in this block bin
                        for t_idx in local_tris_idx
                            tri = mesh.triangles[t_idx]
                            v1 = SVector(tri[1]) .+ mesh_offset
                            v2 = SVector(tri[2]) .+ mesh_offset
                            v3 = SVector(tri[3]) .+ mesh_offset
                            
                            if test_triangle_box_overlap(center_t, half_size_val, v1, v2, v3)
                                obstacles[i, j, k, b] = true
                                # count is not thread safe, skipping exact count for speed
                                break # Marked, move to next voxel
                            end
                        end
                    end
                end
            end
        end
    end
    
    filled = count(obstacles)
    println("[Domain] Marked ~ $filled surface voxels.")
    copyto!(level.obstacle, obstacles)
end

function compute_q_values!(level, mesh::SolverMesh, mesh_offset_t)
    println("[Bouzidi] Computing Q-map for Level $(level.level_id)...")
    
    obstacles = Array(level.obstacle)
    nx, ny, nz = size(obstacles)[1:3]
    active_coords = level.active_block_coords
    n_blocks = length(active_coords)
    dx = level.dx
    mesh_offset = SVector(mesh_offset_t)
    
    # 1. Build Spatial Map (Crucial Optimization)
    println("[Bouzidi] Binning triangles...")
    # Using a slightly larger margin (2.0*dx) to catch rays extending outside block
    tri_map = build_triangle_block_map(mesh, active_coords, dx, mesh_offset, 2.5)
    
    q_map = zeros(Float16, nx, ny, nz, n_blocks, 27)
    
    cx = Int[0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1]
    cy = Int[0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
    cz = Int[0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    
    # Thread-safe counter
    boundary_cells_count = Threads.Atomic{Int}(0)
    
    Base.Threads.@threads for b in 1:n_blocks
        local_tris_idx = tri_map[b]
        # Even if empty, we might need to check if rays go OUT of block into a neighbor's geometry.
        # But for efficiency, we assume triangles relevant to a block are in its bin.
        if isempty(local_tris_idx); continue; end
        
        bx, by, bz = active_coords[b]
        box_min_x = (bx-1)*nx*dx
        box_min_y = (by-1)*ny*dx
        box_min_z = (bz-1)*nz*dx
        
        local_b_count = 0
        
        for z in 1:nz, y in 1:ny, x in 1:nx
            if !obstacles[x,y,z,b]
                is_boundary = false
                
                origin_val = (
                    box_min_x + (x-0.5)*dx + mesh_offset[1],
                    box_min_y + (y-0.5)*dx + mesh_offset[2],
                    box_min_z + (z-0.5)*dx + mesh_offset[3]
                )
                
                for k in 2:27
                    # Only calculate if we are arguably close to a wall?
                    # For now, check all directions. Optimization: check if neighbor is obstacle.
                    
                    dir_val = (Float64(cx[k]), Float64(cy[k]), Float64(cz[k]))
                    len_link = norm(SVector(dir_val)) * dx
                    dir_norm = dir_val ./ norm(SVector(dir_val))
                    
                    min_t = Inf
                    
                    # ITERATE ONLY LOCAL TRIANGLES
                    for t_idx in local_tris_idx
                        tri = mesh.triangles[t_idx]
                        t = intersect_ray_triangle(origin_val, dir_norm, tri[1] .+ mesh_offset_t, tri[2] .+ mesh_offset_t, tri[3] .+ mesh_offset_t)
                        if t < min_t
                            min_t = t
                        end
                    end
                    
                    if min_t <= len_link
                        q = min_t / len_link
                        q_map[x, y, z, b, k] = Float16(q)
                        is_boundary = true
                    end
                end
                if is_boundary
                    local_b_count += 1
                end
            end
        end
        Threads.atomic_add!(boundary_cells_count, local_b_count)
    end
    
    total_boundary = boundary_cells_count[]
    println("[Bouzidi] Computed Q-values. Found $total_boundary boundary cells.")
    copyto!(level.bouzidi_q_map, q_map)
    level.n_boundary_cells = total_boundary
end

function compute_narrow_band_sdf!(level, mesh::SolverMesh, mesh_offset_t)
    println("[Geometry] Computing Narrow-Band SDF for Level $(level.level_id)...")
    
    obstacles = Array(level.obstacle)
    active_coords = level.active_block_coords
    nx, ny, nz = size(obstacles)[1:3]
    n_blocks = length(active_coords)
    dx = level.dx
    mesh_offset = SVector(mesh_offset_t)
    
    # Use the same binning for SDF!
    println("[Geometry] Binning triangles for SDF...")
    tri_map = build_triangle_block_map(mesh, active_coords, dx, mesh_offset, 1.5)
    
    wall_dists = fill(100.0f0, size(obstacles)) 
    count_sdf = Threads.Atomic{Int}(0)
    
    Base.Threads.@threads for b in 1:n_blocks
        local_tris_idx = tri_map[b]
        if isempty(local_tris_idx); continue; end
        
        bx, by, bz = active_coords[b]
        local_count = 0
        
        for z in 1:nz, y in 1:ny, x in 1:nx
            if !obstacles[x, y, z, b]
                # Check neighbors
                is_near = false
                if (x>1 && obstacles[x-1,y,z,b]) || (x<nx && obstacles[x+1,y,z,b]) ||
                   (y>1 && obstacles[x,y-1,z,b]) || (y<ny && obstacles[x,y+1,z,b]) ||
                   (z>1 && obstacles[x,y,z-1,b]) || (z<nz && obstacles[x,y,z+1,b])
                    is_near = true
                end
                
                if is_near
                    gx = (bx-1)*nx + x
                    gy = (by-1)*ny + y
                    gz = (bz-1)*nz + z
                    
                    px = (gx - 0.5) * dx + mesh_offset[1]
                    py = (gy - 0.5) * dx + mesh_offset[2]
                    pz = (gz - 0.5) * dx + mesh_offset[3]
                    P_t = (px, py, pz)
                    
                    min_dist_sq = Inf
                    
                    # CHECK ONLY LOCAL TRIANGLES
                    for t_idx in local_tris_idx
                        tri = mesh.triangles[t_idx]
                        d2 = point_triangle_distance_sq(P_t, tri[1] .+ mesh_offset_t, tri[2] .+ mesh_offset_t, tri[3] .+ mesh_offset_t)
                        if d2 < min_dist_sq
                            min_dist_sq = d2
                        end
                    end
                    
                    wall_dists[x,y,z,b] = Float32(sqrt(min_dist_sq))
                    local_count += 1
                else
                    wall_dists[x,y,z,b] = -1.0f0 
                end
            end
        end
        Threads.atomic_add!(count_sdf, local_count)
    end
    
    c = count_sdf[]
    println("[Geometry] Flagged $c cells for Wall Model.")
    copyto!(level.wall_dist, wall_dists)
end

function point_triangle_distance_sq(P_t, A_t, B_t, C_t)
    P = SVector(P_t)
    A = SVector(A_t)
    B = SVector(B_t)
    C = SVector(C_t)

    e0 = B - A
    e1 = C - A
    v0 = A - P

    a = dot(e0, e0)
    b = dot(e0, e1)
    c = dot(e1, e1)
    d = dot(e0, v0)
    e = dot(e1, v0)

    det = a*c - b*b
    s = b*e - c*d
    t = b*d - a*e

    if (s + t <= det)
        if (s < 0.0)
            if (t < 0.0)  # Region 4
                if (d < 0.0)
                    t = 0.0
                    s = (-d >= a ? 1.0 : -d/a)
                else
                    s = 0.0
                    t = (e >= 0.0 ? 0.0 : (-e >= c ? 1.0 : -e/c))
                end
            else  # Region 3
                s = 0.0
                t = (e >= 0.0 ? 0.0 : (-e >= c ? 1.0 : -e/c))
            end
        elseif (t < 0.0)  # Region 5
            t = 0.0
            s = (d >= 0.0 ? 0.0 : (-d >= a ? 1.0 : -d/a))
        else  # Region 0
            invDet = 1.0 / det
            s *= invDet
            t *= invDet
        end
    else
        if (s < 0.0)  # Region 2
            tmp0 = b + d
            tmp1 = c + e
            if (tmp1 > tmp0)
                numer = tmp1 - tmp0
                denom = a - 2*b + c
                s = (numer >= denom ? 1.0 : numer/denom)
                t = 1.0 - s
            else
                s = 0.0
                t = (tmp1 <= 0.0 ? 1.0 : (e >= 0.0 ? 0.0 : -e/c))
            end
        elseif (t < 0.0)  # Region 6
            tmp0 = b + e
            tmp1 = a + d
            if (tmp1 > tmp0)
                numer = tmp1 - tmp0
                denom = a - 2*b + c
                t = (numer >= denom ? 1.0 : numer/denom)
                s = 1.0 - t
            else
                t = 0.0
                s = (tmp1 <= 0.0 ? 1.0 : (d >= 0.0 ? 0.0 : -d/a))
            end
        else  # Region 1
            numer = c + e - b - d
            denom = a - 2*b + c
            s = (numer >= denom ? 1.0 : numer/denom)
            t = 1.0 - s
        end
    end

    Q = A + s*e0 + t*e1
    diff = P - Q
    return dot(diff, diff)
end

end