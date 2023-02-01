using FFTW
using LinearAlgebra
using Plots

# Using my own local RIDC module
using .RIDC

# Set current directory to be the working directory
cd(@__DIR__)

# PDE
# 2D Incompressible Navier Stokes (constant density ρ₀)
# ∂u/∂t + (u·∇)u - μ/ρ₀ ∇²u = -∇(p/ρ₀) + f

# 2D Vorticity Equation (kinematic viscosity v)
# ∂w/∂t = v ∇²w - (u·∇)w + ∇ x f

# Spatial Domain
# (x, y) ∈ {(x, y) : -π <= x <= π, -π <= y <= π}

# Time Domain
# t ∈ [0, ∞)

# Initial Condition
# u(x, y, 0) = 0    Fluid is initially at rest

# Boundary Conditions
# Periodic boundary with period 2π

# Pseudo-Spectral Method Solution (Euler Method Time Integration)

N_POINTS = 100
KINEMATIC_VISCOSITY = 0.5
MAX_TIME = 2.0
N_TIME_NODES = 1000
N_TIME_INTERVAL_NODES = 100
N_CORRECTIONS = 2

function main()
    element_length = 2*pi / (N_POINTS - 1)

    x_range = range(-pi, pi, length = N_POINTS)
    y_range = range(-pi, pi, length = N_POINTS)

    # Get discretised spatial grid
    coordinates_x = [x for x in x_range, y in y_range]
    coordinates_y = [y for x in x_range, y in y_range]

    # Get wavenumbers in spatial grid format and their norms
    wavenumbers_1d = fftfreq(N_POINTS) .* N_POINTS
    wavenumbers_x = [kx for kx in wavenumbers_1d, ky in wavenumbers_1d]
    wavenumbers_y = [ky for kx in wavenumbers_1d, ky in wavenumbers_1d]
    wavenumbers_norm = [norm([kx, ky]) for kx in wavenumbers_1d, ky in wavenumbers_1d]

    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1
    wavenumbers_normalised_x = wavenumbers_x ./ wavenumbers_norm
    wavenumbers_normalised_y = wavenumbers_y ./ wavenumbers_norm

    # Define body force which acts only in x direction
    force_x = 10 * (
        exp.(-5 * ((coordinates_x .+ 0.5*pi).^2 + (coordinates_y .+ 0.2*pi).^2))
        -
        exp.(-5 * ((coordinates_x .- 0.5*pi).^2 + (coordinates_y .- 0.2*pi).^2))
    )

    # Get fourier transform of body force
    force_x_fft = fft(force_x)

    # Define RHS of the ODE for the spatial fourier transformed vorticity
    function f(w_fft, t)
        w_fft = reshape(w_fft, N_POINTS, N_POINTS)

        # Initialise time derivative of the fourier transformed vorticity for current time step
        dw_dt_fft = zero(coordinates_x)

        # Get velocity components at current time
        velo_x =  ifft(1im * w_fft .* wavenumbers_y ./ wavenumbers_norm)
        velo_y = -ifft(1im * w_fft .* wavenumbers_x ./ wavenumbers_norm)

        # Apply convection term in fourier space
        dw_dt_fft -= fft(
            velo_x .* ifft(1im * w_fft .* wavenumbers_x)
            +
            velo_y .* ifft(1im * w_fft .* wavenumbers_y)
        )

        # Apply a mask which sets to 0 the values of the non-linear convection term with large wavenumbers
        mask_wavenumbers = (abs.(wavenumbers_1d) .>= 2*fld(N_POINTS, 3))
        dw_dt_fft[mask_wavenumbers, mask_wavenumbers] .= 0

        # Apply curl of body force term in fourier space for a unit of time only
        pre_factor = max(1 - t, 0)
        dw_dt_fft -= pre_factor * 1im * wavenumbers_y .* force_x_fft

        # Apply diffusion term in fourier space
        dw_dt_fft -= KINEMATIC_VISCOSITY * wavenumbers_norm .* w_fft

        return reshape(dw_dt_fft, N_POINTS^2)
    end

    # The fluid is initially at rest
    u0 = zeros(ComplexF64, N_POINTS^2)
    T = MAX_TIME
    N = N_TIME_NODES
    M = N_CORRECTIONS
    K = N_TIME_INTERVAL_NODES

    # Use the RIDC2 method to approximately solve the ODE with order of accuracy
    # with respect to time of 2(M+1)
    t, w_fft = RIDC2(f, u0, T, N, M, K)

    # Convert the spatially fourier transformed vorticities into normal space
    w = reshape(ifft(w_fft), :, N_POINTS, N_POINTS)

    # Initialise animation of vorticity
    anim = Plots.Animation()

    for i in 1:N_TIME_NODES+1
        frame(
            anim, 
            heatmap(
                x_range, 
                y_range, 
                w[i]', 
                c = :diverging_bkr_55_10_c35_n256, 
                aspect_ratio = :equal, 
                size = (680, 650)
            )
        )
    end
    display(gif(anim, "2D-Navier-Stokes-RIDC2.gif", fps = 2))
end

main()