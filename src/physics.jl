# FILE: .\src\physics.jl
# 
using KernelAbstractions
using StaticArrays
using CUDA
using Adapt

const KAPPA = 0.41f0
const E_CONST = 9.8f0
const CS2 = 1.0f0 / 3.0f0
const A_PLUS = 26.0f0

@inline function gpu_hash(x::Int32)::UInt32
    h = reinterpret(UInt32, x)
    h = (h ⊻ (h >> 16)) * UInt32(0x85ebca6b)
    h = (h ⊻ (h >> 13)) * UInt32(0xc2b2ae35)
    h = h ⊻ (h >> 16)
    return h
end

@inline function gradient_noise(gx::Int32, gy::Int32, gz::Int32, seed::Int32)::Float32
    combined = gx * Int32(374761393) + gy * Int32(668265263) + gz * Int32(1274126177) + seed
    h = gpu_hash(combined)
    return (Float32(h & UInt32(0xFFFF)) / 32768.0f0) - 1.0f0
end

"""
    get_velocity_neighbor

Retrieves velocity from a neighbor cell. 
Types are left generic to allow the compiler to inline and specialize 
regardless of whether indices are Int32 or Int64.
"""
@inline function get_velocity_neighbor(
    vel_in,
    x, y, z, b_idx,
    dx, dy, dz,
    block_size,
    neighbor_table
)
    
    ix = Int32(x)
    iy = Int32(y)
    iz = Int32(z)
    ib_idx = Int32(b_idx)
    idx = Int32(dx)
    idy = Int32(dy)
    idz = Int32(dz)
    ibs = Int32(block_size)

    nx = ix + idx
    ny = iy + idy
    nz = iz + idz

    
    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        vx = vel_in[nx, ny, nz, ib_idx, 1]
        vy = vel_in[nx, ny, nz, ib_idx, 2]
        vz = vel_in[nx, ny, nz, ib_idx, 3]
        return vx, vy, vz
    end

    
    off_x = (nx < Int32(1)) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = (ny < Int32(1)) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = (nz < Int32(1)) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))

    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1)) * Int32(3) + (off_z + Int32(1)) * Int32(9) + Int32(1)
    nb_idx_global = neighbor_table[ib_idx, dir_idx]

    if nb_idx_global > Int32(0)
        
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        
        vx = vel_in[nnx, nny, nnz, nb_idx_global, 1]
        vy = vel_in[nnx, nny, nnz, nb_idx_global, 2]
        vz = vel_in[nnx, nny, nnz, nb_idx_global, 3]
        return vx, vy, vz
    end

    
    return vel_in[ix, iy, iz, ib_idx, 1], vel_in[ix, iy, iz, ib_idx, 2], vel_in[ix, iy, iz, ib_idx, 3]
end

"""
    stream_collide_kernel!

Standard A-B Pattern LBM Kernel with WALE Turbulence Model.
Reads f_in/vel_in, Collides, Writes f_out/vel_out.
"""
@kernel function stream_collide_kernel!(
    f_out, f_in,                      
    rho_out, vel_out, vel_in,
    obstacle, sponge_arr,
    wall_dist_arr,
    neighbor_table,
    active_coords_x, active_coords_y, active_coords_z,
    parent_f, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    tau_molecular::Float32, c_wale::Float32,
    cs2_val::Float32, cs4_val::Float32,
    is_level_1::Bool,
    is_symmetric::Bool,
    nx_global::Int32, ny_global::Int32, nz_global::Int32, 
    u_inlet::Float32,
    n_blocks::Int32,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr, opp_arr,
    mirror_y_arr, mirror_z_arr,
    wall_model_active::Bool,
    time_step_seed::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)
    
    if b_idx <= n_blocks
        @inbounds begin
            bx = active_coords_x[b_idx]
            by = active_coords_y[b_idx]
            bz = active_coords_z[b_idx]
            gx = (bx - Int32(1)) * block_size + Int32(x)
            gy = (by - Int32(1)) * block_size + Int32(y)
            gz = (bz - Int32(1)) * block_size + Int32(z)
            
            is_obs = obstacle[x, y, z, b_idx]
            
            rho = 0.0f0
            jx = 0.0f0
            jy = 0.0f0
            jz = 0.0f0
            
            f_local_1=0.0f0; f_local_2=0.0f0; f_local_3=0.0f0; f_local_4=0.0f0; f_local_5=0.0f0
            f_local_6=0.0f0; f_local_7=0.0f0; f_local_8=0.0f0; f_local_9=0.0f0; f_local_10=0.0f0
            f_local_11=0.0f0; f_local_12=0.0f0; f_local_13=0.0f0; f_local_14=0.0f0; f_local_15=0.0f0
            f_local_16=0.0f0; f_local_17=0.0f0; f_local_18=0.0f0; f_local_19=0.0f0; f_local_20=0.0f0
            f_local_21=0.0f0; f_local_22=0.0f0; f_local_23=0.0f0; f_local_24=0.0f0; f_local_25=0.0f0
            f_local_26=0.0f0; f_local_27=0.0f0

            
            
            
            for k in Int32(1):Int32(27)
                cx = cx_arr[k]
                cy = cy_arr[k]
                cz = cz_arr[k]
                
                sx = Int32(x) - cx
                sy = Int32(y) - cy
                sz = Int32(z) - cz
                
                val = 0.0f0
                
                if sx >= Int32(1) && sx <= block_size && sy >= Int32(1) && sy <= block_size && sz >= Int32(1) && sz <= block_size
                    val = f_in[sx, sy, sz, b_idx, k]
                else
                    nb_off_x = (sx < Int32(1)) ? Int32(-1) : (sx > block_size ? Int32(1) : Int32(0))
                    nb_off_y = (sy < Int32(1)) ? Int32(-1) : (sy > block_size ? Int32(1) : Int32(0))
                    nb_off_z = (sz < Int32(1)) ? Int32(-1) : (sz > block_size ? Int32(1) : Int32(0))
                    dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) + (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
                    nb_idx_global = neighbor_table[b_idx, dir_idx]
                    
                    if nb_idx_global > Int32(0)
                        nsx = sx < Int32(1) ? sx + block_size : (sx > block_size ? sx - block_size : sx)
                        nsy = sy < Int32(1) ? sy + block_size : (sy > block_size ? sy - block_size : sy)
                        nsz = sz < Int32(1) ? sz + block_size : (sz > block_size ? sz - block_size : sz)
                        val = f_in[nsx, nsy, nsz, nb_idx_global, k]
                    else
                        src_gx = gx - cx
                        src_gy = gy - cy
                        src_gz = gz - cz
                        
                        is_inlet = (src_gx < Int32(1))
                        is_outlet = (src_gx > nx_global)
                        is_y_min = (src_gy < Int32(1))
                        is_y_max = (src_gy > ny_global)
                        is_z_min = (src_gz < Int32(1))
                        is_z_max = (src_gz > nz_global)
                        
                        if is_inlet 
                            cx_f = Float32(cx)
                            turb_intensity = 0.02f0
                            noise = gradient_noise(gy, gz, time_step_seed, Int32(1234)) * turb_intensity * u_inlet
                            u_inst = u_inlet + noise
                            
                            cu_in = cx_f * u_inst
                            usq_in = u_inst * u_inst
                            val = w_arr[k] * (1.0f0 + 3.0f0*cu_in + 4.5f0*cu_in*cu_in - 1.5f0*usq_in)
                            
                        elseif is_outlet
                            if cx < Int32(0)
                                val = w_arr[k] * (1.0f0 + 3.0f0*Float32(cx)*u_inlet)
                            else
                                val = f_in[x, y, z, b_idx, k]
                            end
                        
                        # CHANGED: Added mirror condition for Y boundaries regardless of symmetry flag.
                        # This ensures lateral walls act as Slip Walls instead of vacuum.
                        elseif (is_y_min || is_y_max) 
                            mirror_k = mirror_y_arr[k]
                            val = f_in[x, y, z, b_idx, mirror_k]
                            
                        elseif is_z_min || is_z_max
                            # Slip walls for top/bottom
                            mirror_k = mirror_z_arr[k]
                            val = f_in[x, y, z, b_idx, mirror_k]
                            
                        elseif !is_level_1
                            pgx = (src_gx + Int32(1)) ÷ Int32(2)
                            pgy = (src_gy + Int32(1)) ÷ Int32(2)
                            pgz = (src_gz + Int32(1)) ÷ Int32(2)
                            pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
                            pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
                            pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
                            
                            if pbx >= Int32(1) && pbx <= parent_dim_x && 
                               pby >= Int32(1) && pby <= parent_dim_y && 
                               pbz >= Int32(1) && pbz <= parent_dim_z
                                pb_idx = parent_ptr[pbx, pby, pbz]
                                if pb_idx > Int32(0)
                                    plx = (pgx - Int32(1)) % block_size + Int32(1)
                                    ply = (pgy - Int32(1)) % block_size + Int32(1)
                                    plz = (pgz - Int32(1)) % block_size + Int32(1)
                                    val = parent_f[plx, ply, plz, pb_idx, k]
                                else
                                    val = w_arr[k]
                                end
                            else
                                val = w_arr[k]
                            end
                        else
                            val = w_arr[k]
                        end
                    end
                end
                
                
                if k == Int32(1); f_local_1=val
                elseif k == Int32(2); f_local_2=val
                elseif k == Int32(3); f_local_3=val
                elseif k == Int32(4); f_local_4=val
                elseif k == Int32(5); f_local_5=val
                elseif k == Int32(6); f_local_6=val
                elseif k == Int32(7); f_local_7=val
                elseif k == Int32(8); f_local_8=val
                elseif k == Int32(9); f_local_9=val
                elseif k == Int32(10); f_local_10=val
                elseif k == Int32(11); f_local_11=val
                elseif k == Int32(12); f_local_12=val
                elseif k == Int32(13); f_local_13=val
                elseif k == Int32(14); f_local_14=val
                elseif k == Int32(15); f_local_15=val
                elseif k == Int32(16); f_local_16=val
                elseif k == Int32(17); f_local_17=val
                elseif k == Int32(18); f_local_18=val
                elseif k == Int32(19); f_local_19=val
                elseif k == Int32(20); f_local_20=val
                elseif k == Int32(21); f_local_21=val
                elseif k == Int32(22); f_local_22=val
                elseif k == Int32(23); f_local_23=val
                elseif k == Int32(24); f_local_24=val
                elseif k == Int32(25); f_local_25=val
                elseif k == Int32(26); f_local_26=val
                elseif k == Int32(27); f_local_27=val
                end
                
                rho += val
                jx += val * Float32(cx_arr[k])
                jy += val * Float32(cy_arr[k])
                jz += val * Float32(cz_arr[k])
            end

            
            
            
            
            if is_obs
                
                vel_out[x, y, z, b_idx, 1] = 0.0f0
                vel_out[x, y, z, b_idx, 2] = 0.0f0
                vel_out[x, y, z, b_idx, 3] = 0.0f0
                rho_out[x, y, z, b_idx] = 1.0f0
                
                for k in Int32(1):Int32(27)
                    opp = opp_arr[k]
                    val_out = 0.0f0
                    if opp == Int32(1); val_out=f_local_1
                    elseif opp == Int32(2); val_out=f_local_2
                    elseif opp == Int32(3); val_out=f_local_3
                    elseif opp == Int32(4); val_out=f_local_4
                    elseif opp == Int32(5); val_out=f_local_5
                    elseif opp == Int32(6); val_out=f_local_6
                    elseif opp == Int32(7); val_out=f_local_7
                    elseif opp == Int32(8); val_out=f_local_8
                    elseif opp == Int32(9); val_out=f_local_9
                    elseif opp == Int32(10); val_out=f_local_10
                    elseif opp == Int32(11); val_out=f_local_11
                    elseif opp == Int32(12); val_out=f_local_12
                    elseif opp == Int32(13); val_out=f_local_13
                    elseif opp == Int32(14); val_out=f_local_14
                    elseif opp == Int32(15); val_out=f_local_15
                    elseif opp == Int32(16); val_out=f_local_16
                    elseif opp == Int32(17); val_out=f_local_17
                    elseif opp == Int32(18); val_out=f_local_18
                    elseif opp == Int32(19); val_out=f_local_19
                    elseif opp == Int32(20); val_out=f_local_20
                    elseif opp == Int32(21); val_out=f_local_21
                    elseif opp == Int32(22); val_out=f_local_22
                    elseif opp == Int32(23); val_out=f_local_23
                    elseif opp == Int32(24); val_out=f_local_24
                    elseif opp == Int32(25); val_out=f_local_25
                    elseif opp == Int32(26); val_out=f_local_26
                    elseif opp == Int32(27); val_out=f_local_27
                    end
                    
                    f_out[x, y, z, b_idx, k] = val_out
                end
            else
                
                rho = max(rho, 0.01f0)
                inv_rho = 1.0f0 / rho
                ux = jx * inv_rho
                uy = jy * inv_rho
                uz = jz * inv_rho
                
                
                sp = sponge_arr[x, y, z, b_idx]
                if sp > 0.0f0
                    rho = rho * (1.0f0 - sp) + 1.0f0 * sp
                    ux = ux * (1.0f0 - sp) + u_inlet * sp
                    uy = uy * (1.0f0 - sp)
                    uz = uz * (1.0f0 - sp)
                end
                
                
                v_mag_sq = ux*ux + uy*uy + uz*uz
                if v_mag_sq > 0.30f0
                    scale = sqrt(0.30f0 / v_mag_sq)
                    ux = ux * scale
                    uy = uy * scale
                    uz = uz * scale
                    v_mag_sq = 0.30f0
                end
                
                
                Fx_wall = 0.0f0
                Fy_wall = 0.0f0
                Fz_wall = 0.0f0
                y_plus = 1000.0f0
                
                if wall_model_active
                    dist_wall = wall_dist_arr[x, y, z, b_idx]
                    
                    if dist_wall > 0.0f0 && dist_wall < 10.0f0
                        u_mag = sqrt(v_mag_sq)
                        nu_visc = (tau_molecular - 0.5f0) / 3.0f0
                        
                        if u_mag > 1.0f-6 && nu_visc > 1.0f-10
                            A_ww = 8.3f0
                            B_ww = 1.0f0 / 7.0f0
                            y_plus_crit = 11.81f0
                            
                            u_tau = u_mag * (nu_visc / (dist_wall * u_mag + 1.0f-10))^B_ww * (2.0f0 * A_ww)^(-B_ww)
                            u_tau = max(u_tau, 1.0f-6)
                            
                            y_p = u_tau * dist_wall / nu_visc
                            if y_p > y_plus_crit
                                u_plus_target = u_mag / u_tau
                                u_plus_law = (1.0f0 / KAPPA) * log(y_p) + 5.2f0
                                
                                if u_plus_law > 0.1f0
                                    u_tau = u_tau * (u_plus_target / u_plus_law)
                                    u_tau = max(u_tau, 1.0f-6)
                                end
                            end
                            
                            y_plus = u_tau * dist_wall / nu_visc
                            
                            tau_wall = rho * u_tau * u_tau
                            tau_res = rho * nu_visc * (u_mag / dist_wall)
                            
                            if tau_wall > tau_res && u_mag > 1.0f-6
                                force_mag = (tau_wall - tau_res) / dist_wall
                                Fx_wall = -force_mag * ux / u_mag
                                Fy_wall = -force_mag * uy / u_mag
                                Fz_wall = -force_mag * uz / u_mag
                            end
                        end
                    end
                end
                
                ux_eq = ux + 0.5f0 * Fx_wall * inv_rho
                uy_eq = uy + 0.5f0 * Fy_wall * inv_rho
                uz_eq = uz + 0.5f0 * Fz_wall * inv_rho
                usq_eq = ux_eq*ux_eq + uy_eq*uy_eq + uz_eq*uz_eq
                
                vel_out[x, y, z, b_idx, 1] = ux
                vel_out[x, y, z, b_idx, 2] = uy
                vel_out[x, y, z, b_idx, 3] = uz
                rho_out[x, y, z, b_idx] = rho
                
                
                Pxx = 0.0f0; Pyy = 0.0f0; Pzz = 0.0f0
                Pxy = 0.0f0; Pxz = 0.0f0; Pyz = 0.0f0
                
                for k in Int32(1):Int32(27)
                    val = 0.0f0
                    if k == Int32(1); val = f_local_1
                    elseif k == Int32(2); val = f_local_2
                    elseif k == Int32(3); val = f_local_3
                    elseif k == Int32(4); val = f_local_4
                    elseif k == Int32(5); val = f_local_5
                    elseif k == Int32(6); val = f_local_6
                    elseif k == Int32(7); val = f_local_7
                    elseif k == Int32(8); val = f_local_8
                    elseif k == Int32(9); val = f_local_9
                    elseif k == Int32(10); val = f_local_10
                    elseif k == Int32(11); val = f_local_11
                    elseif k == Int32(12); val = f_local_12
                    elseif k == Int32(13); val = f_local_13
                    elseif k == Int32(14); val = f_local_14
                    elseif k == Int32(15); val = f_local_15
                    elseif k == Int32(16); val = f_local_16
                    elseif k == Int32(17); val = f_local_17
                    elseif k == Int32(18); val = f_local_18
                    elseif k == Int32(19); val = f_local_19
                    elseif k == Int32(20); val = f_local_20
                    elseif k == Int32(21); val = f_local_21
                    elseif k == Int32(22); val = f_local_22
                    elseif k == Int32(23); val = f_local_23
                    elseif k == Int32(24); val = f_local_24
                    elseif k == Int32(25); val = f_local_25
                    elseif k == Int32(26); val = f_local_26
                    elseif k == Int32(27); val = f_local_27
                    end
                    
                    cx_f = Float32(cx_arr[k])
                    cy_f = Float32(cy_arr[k])
                    cz_f = Float32(cz_arr[k])
                    
                    Pxx += val * cx_f * cx_f
                    Pyy += val * cy_f * cy_f
                    Pzz += val * cz_f * cz_f
                    Pxy += val * cx_f * cy_f
                    Pxz += val * cx_f * cz_f
                    Pyz += val * cy_f * cz_f
                end
                
                Qxx = Pxx - (rho*ux*ux + rho*CS2)
                Qyy = Pyy - (rho*uy*uy + rho*CS2)
                Qzz = Pzz - (rho*uz*uz + rho*CS2)
                Qxy = Pxy - rho*ux*uy
                Qxz = Pxz - rho*ux*uz
                Qyz = Pyz - rho*uy*uz
                
                
                
                P_neq = sqrt(Qxx*Qxx + Qyy*Qyy + Qzz*Qzz + 2.0f0*(Qxy*Qxy + Qxz*Qxz + Qyz*Qyz))
                
                
                
                
                
                
                
                
                
                ux_E, uy_E, uz_E = get_velocity_neighbor(vel_in, x, y, z, b_idx, Int32(1), Int32(0), Int32(0), block_size, neighbor_table)
                ux_W, uy_W, uz_W = get_velocity_neighbor(vel_in, x, y, z, b_idx, Int32(-1), Int32(0), Int32(0), block_size, neighbor_table)
                ux_N, uy_N, uz_N = get_velocity_neighbor(vel_in, x, y, z, b_idx, Int32(0), Int32(1), Int32(0), block_size, neighbor_table)
                ux_S, uy_S, uz_S = get_velocity_neighbor(vel_in, x, y, z, b_idx, Int32(0), Int32(-1), Int32(0), block_size, neighbor_table)
                ux_T, uy_T, uz_T = get_velocity_neighbor(vel_in, x, y, z, b_idx, Int32(0), Int32(0), Int32(1), block_size, neighbor_table)
                ux_B, uy_B, uz_B = get_velocity_neighbor(vel_in, x, y, z, b_idx, Int32(0), Int32(0), Int32(-1), block_size, neighbor_table)

                
                
                
                g11 = 0.5f0 * (ux_E - ux_W) 
                g12 = 0.5f0 * (ux_N - ux_S) 
                g13 = 0.5f0 * (ux_T - ux_B) 
                
                g21 = 0.5f0 * (uy_E - uy_W) 
                g22 = 0.5f0 * (uy_N - uy_S) 
                g23 = 0.5f0 * (uy_T - uy_B) 
                
                g31 = 0.5f0 * (uz_E - uz_W) 
                g32 = 0.5f0 * (uz_N - uz_S) 
                g33 = 0.5f0 * (uz_T - uz_B) 
                
                
                gsq11 = g11*g11 + g12*g21 + g13*g31
                gsq12 = g11*g12 + g12*g22 + g13*g32
                gsq13 = g11*g13 + g12*g23 + g13*g33
                
                gsq21 = g21*g11 + g22*g21 + g23*g31
                gsq22 = g21*g12 + g22*g22 + g23*g32
                gsq23 = g21*g13 + g22*g23 + g23*g33
                
                gsq31 = g31*g11 + g32*g21 + g33*g31
                gsq32 = g31*g12 + g32*g22 + g33*g32
                gsq33 = g31*g13 + g32*g23 + g33*g33
                
                
                tr_gsq = gsq11 + gsq22 + gsq33
                tr_term = tr_gsq / 3.0f0
                
                
                Sd11 = 0.5f0 * (gsq11 + gsq11) - tr_term
                Sd22 = 0.5f0 * (gsq22 + gsq22) - tr_term
                Sd33 = 0.5f0 * (gsq33 + gsq33) - tr_term
                Sd12 = 0.5f0 * (gsq12 + gsq21)
                Sd13 = 0.5f0 * (gsq13 + gsq31)
                Sd23 = 0.5f0 * (gsq23 + gsq32)
                
                
                S11 = g11
                S22 = g22
                S33 = g33
                S12 = 0.5f0 * (g12 + g21)
                S13 = 0.5f0 * (g13 + g31)
                S23 = 0.5f0 * (g23 + g32)
                
                
                
                OP1 = Sd11*Sd11 + Sd22*Sd22 + Sd33*Sd33 + 2.0f0*(Sd12*Sd12 + Sd13*Sd13 + Sd23*Sd23)
                
                
                OP2 = S11*S11 + S22*S22 + S33*S33 + 2.0f0*(S12*S12 + S13*S13 + S23*S23)
                
                nu_eddy = 0.0f0
                if OP1 > 1.0f-10
                    
                    
                    numerator = OP1 * sqrt(OP1)
                    denom = (OP2 * OP2 * sqrt(OP2)) + (OP1 * sqrt(sqrt(OP1)))
                    
                    nu_eddy = (c_wale * c_wale) * numerator / denom
                end
                
                
                
                
                tau_turb = tau_molecular + nu_eddy * 3.0f0
                
                omega = 1.0f0 / max(tau_turb, 0.500001f0)
                
                
                
                
                
                
                for k in Int32(1):Int32(27)
                    cx_f = Float32(cx_arr[k])
                    cy_f = Float32(cy_arr[k])
                    cz_f = Float32(cz_arr[k])
                    w_k = w_arr[k]
                    
                    cu = cx_f*ux_eq + cy_f*uy_eq + cz_f*uz_eq
                    feq = rho * w_k * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq_eq)
                    
                    
                    xi_x = cx_f - ux
                    xi_y = cy_f - uy
                    xi_z = cz_f - uz
                    proj_neq = (w_k * 4.5f0) * (
                        Qxx*(xi_x*xi_x - CS2) + Qyy*(xi_y*xi_y - CS2) + Qzz*(xi_z*xi_z - CS2) + 
                        2.0f0*(Qxy*xi_x*xi_y + Qxz*xi_x*xi_z + Qyz*xi_y*xi_z)
                    )
                    
                    force_term = w_k * 3.0f0 * (cx_f*Fx_wall + cy_f*Fy_wall + cz_f*Fz_wall)
                    
                    f_out[x, y, z, b_idx, k] = feq + (1.0f0 - omega)*proj_neq + (1.0f0 - 0.5f0*omega)*force_term
                end
            end
        end
    end
end

"""
    perform_timestep!

executes one timestep using A-B pattern for populations (reading f_in, writing f_out)
and Ping-Pong pattern for velocity (reading vel_in, writing vel_out) for WALE gradients.
"""
function perform_timestep!(
    level, 
    parent_f, parent_ptr, 
    f_out, f_in,
    vel_out, vel_in,
    u_curr,
    cx_gpu, cy_gpu, cz_gpu, w_gpu, 
    opp_gpu, mirror_y_gpu, mirror_z_gpu,
    domain_nx, domain_ny, domain_nz,
    wall_model_active::Bool,
    c_wale_val::Float32,
    timestep::Int
)
    backend = get_backend(f_in)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0
        return
    end
    
    is_l1 = (parent_f === nothing)
    
    if is_l1
        p_f = f_in 
        p_ptr = level.block_pointer 
        px = Int32(1)
        py = Int32(1)
        pz = Int32(1)
    else
        p_f = parent_f
        p_ptr = parent_ptr
        
        px = Int32(size(p_ptr, 1))
        py = Int32(size(p_ptr, 2))
        pz = Int32(size(p_ptr, 3))
    end
    
    scale = 1 << (level.level_id - 1)
    nx_g = Int32(domain_nx * scale)
    ny_g = Int32(domain_ny * scale)
    nz_g = Int32(domain_nz * scale)
    
    ts_seed = Int32(trunc(u_curr * 10000.0f0) % 1000000)

    kernel! = stream_collide_kernel!(backend)
    kernel!(
        f_out, f_in,
        level.rho, vel_out, vel_in,
        level.obstacle, level.sponge,
        level.wall_dist,
        level.neighbor_table, 
        level.map_x, level.map_y, level.map_z,
        p_f, p_ptr,
        px, py, pz,
        level.tau, c_wale_val,
        CS2, CS2*CS2,
        is_l1, SYMMETRIC_ANALYSIS,
        nx_g, ny_g, nz_g,
        u_curr,
        Int32(n_blocks), Int32(BLOCK_SIZE),
        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
        wall_model_active,
        ts_seed,
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    
    if level.bouzidi_enabled && level.n_boundary_cells > 0
        apply_bouzidi_correction!(
            f_out, f_in,
            level.bouzidi_q_map,
            level.bouzidi_cell_block,
            level.bouzidi_cell_x,
            level.bouzidi_cell_y,
            level.bouzidi_cell_z,
            level.n_boundary_cells,
            level.neighbor_table, BLOCK_SIZE,
            cx_gpu, cy_gpu, cz_gpu, opp_gpu,
            Q_MIN_THRESHOLD,
            backend
        )
    end
end

function build_lattice_arrays_gpu(backend)
    cx = Int32[]
    cy = Int32[]
    cz = Int32[]
    w = Float32[]
    
    for dz in -1:1, dy in -1:1, dx in -1:1
        push!(cx, Int32(dx))
        push!(cy, Int32(dy))
        push!(cz, Int32(dz))
        d2 = dx^2 + dy^2 + dz^2
        push!(w, d2==0 ? Float32(8/27) : d2==1 ? Float32(2/27) : d2==2 ? Float32(1/54) : Float32(1/216))
    end
    
    opp = zeros(Int32, 27)
    mirror_y = zeros(Int32, 27)
    mirror_z = zeros(Int32, 27)
    
    for i in 1:27
        for j in 1:27
            if cx[j]==-cx[i] && cy[j]==-cy[i] && cz[j]==-cz[i]
                opp[i] = Int32(j)
            end
            if cx[j]==cx[i] && cy[j]==-cy[i] && cz[j]==cz[i]
                mirror_y[i] = Int32(j)
            end
            if cx[j]==cx[i] && cy[j]==cy[i] && cz[j]==-cz[i]
                mirror_z[i] = Int32(j)
            end
        end
    end
    
    return (
        adapt(backend, cx), 
        adapt(backend, cy), 
        adapt(backend, cz), 
        adapt(backend, w), 
        adapt(backend, opp), 
        adapt(backend, mirror_y), 
        adapt(backend, mirror_z)
    )
end