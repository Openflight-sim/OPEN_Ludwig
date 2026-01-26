// # FILE: .\src\blocks.jl
using KernelAbstractions
using CUDA
using Adapt
using StaticArrays

const BLOCK_SIZE = 8

"""
    BlockLevel

Represents one refinement level of the sparse voxel grid.
Standard A-B Pattern: Uses 'f' and 'f_temp' for safe stream-collide.
Added 'vel_temp' for WALE gradient computation (prevents race conditions).
"""
mutable struct BlockLevel{T_P4, T_P5, T_BlockPtr, T_NbTable, T_Obst, T_F16, T_I8, T_IntVec, T_MapVec}
    level_id::Int
    dx::Float64
    dt::Float32
    tau::Float32
    
    grid_dim_x::Int
    grid_dim_y::Int
    grid_dim_z::Int
    
    block_pointer::T_BlockPtr            
    active_block_coords::Vector{Tuple{Int, Int, Int}}  
    
    rho::T_P4                             
    vel::T_P5                             
    vel_temp::T_P5                        
    
    f::T_P5                               
    f_temp::T_P5                          
    
    wall_dist::T_P4                       
    obstacle::T_Obst                      
    sponge::T_P4                          
    
    neighbor_table::T_NbTable             
    map_x::T_MapVec                       
    map_y::T_MapVec
    map_z::T_MapVec
    
    bouzidi_enabled::Bool
    bouzidi_q_map::T_F16                 
    bouzidi_cell_block::T_IntVec         
    bouzidi_cell_x::T_I8                 
    bouzidi_cell_y::T_I8
    bouzidi_cell_z::T_I8
    n_boundary_cells::Int
end

function Adapt.adapt_structure(to, level::BlockLevel)
    BlockLevel(
        level.level_id,
        level.dx,
        level.dt,
        level.tau,
        level.grid_dim_x,
        level.grid_dim_y,
        level.grid_dim_z,
        adapt(to, level.block_pointer),
        level.active_block_coords,  
        adapt(to, level.rho),
        adapt(to, level.vel),
        adapt(to, level.vel_temp),
        adapt(to, level.f),          
        adapt(to, level.f_temp),     
        adapt(to, level.wall_dist),
        adapt(to, level.obstacle),
        adapt(to, level.sponge),
        adapt(to, level.neighbor_table),
        adapt(to, level.map_x),
        adapt(to, level.map_y),
        adapt(to, level.map_z),
        level.bouzidi_enabled,
        adapt(to, level.bouzidi_q_map),
        adapt(to, level.bouzidi_cell_block),
        adapt(to, level.bouzidi_cell_x),
        adapt(to, level.bouzidi_cell_y),
        adapt(to, level.bouzidi_cell_z),
        level.n_boundary_cells
    )
end

"""
    BlockLevel Constructor

Allocates dual distribution arrays (f, f_temp) and dual velocity arrays (vel, vel_temp).
"""
function BlockLevel(level_id::Int, 
                    active_coords::Vector{Tuple{Int, Int, Int}}, 
                    neighbor_table::Matrix{Int32}, 
                    dx::AbstractFloat, 
                    dt::AbstractFloat, 
                    tau::AbstractFloat;
                    bouzidi_q_map = nothing,
                    bouzidi_cell_block = nothing,
                    bouzidi_cell_x = nothing,
                    bouzidi_cell_y = nothing,
                    bouzidi_cell_z = nothing,
                    n_boundary_cells = 0)

    n_blocks = length(active_coords)
    
    if isempty(active_coords)
        bx_max, by_max, bz_max = 0, 0, 0
    else
        bx_max = maximum(c[1] for c in active_coords)
        by_max = maximum(c[2] for c in active_coords)
        bz_max = maximum(c[3] for c in active_coords)
    end
    
    block_pointer = fill(Int32(0), bx_max, by_max, bz_max)
    for (i, (bx, by, bz)) in enumerate(active_coords)
        block_pointer[bx, by, bz] = Int32(i)
    end

    rho = ones(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    vel = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    vel_temp = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    
    
    f = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
    f_temp = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
    
    wall_dist = fill(100.0f0, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    obstacle  = zeros(Bool, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    sponge    = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    
    map_x = Int32[c[1] for c in active_coords]
    map_y = Int32[c[2] for c in active_coords]
    map_z = Int32[c[3] for c in active_coords]
    
    bouzidi_enabled = (n_boundary_cells > 0) && (bouzidi_q_map !== nothing)
    
    if bouzidi_enabled
        q_map_final = convert(Array{Float16, 5}, bouzidi_q_map)
        c_block_final = convert(Vector{Int32}, bouzidi_cell_block)
        c_x_final = convert(Vector{Int8}, bouzidi_cell_x)
        c_y_final = convert(Vector{Int8}, bouzidi_cell_y)
        c_z_final = convert(Vector{Int8}, bouzidi_cell_z)
    else
        q_map_final = zeros(Float16, 0, 0, 0, 0, 0)
        c_block_final = Int32[]
        c_x_final = Int8[]
        c_y_final = Int8[]
        c_z_final = Int8[]
    end

    return BlockLevel(
        level_id,
        Float64(dx),
        Float32(dt),
        Float32(tau),
        bx_max,            
        by_max,            
        bz_max,            
        block_pointer,
        active_coords,     
        rho,
        vel,
        vel_temp,
        f,                 
        f_temp,
        wall_dist,
        obstacle,
        sponge,
        neighbor_table,
        map_x,
        map_y,
        map_z,
        bouzidi_enabled,
        q_map_final,
        c_block_final,
        c_x_final,
        c_y_final,
        c_z_final,
        n_boundary_cells
    )
end

"""
    SolverMesh (Fluid Grid)

Holds the multi-level grid structure. Distinct from Geometry.SolverMesh.
"""
mutable struct SolverMesh
    levels::Vector{BlockLevel}
    domain_dims::Tuple{Int, Int, Int}
end