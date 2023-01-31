using DataFrames
using CSV

# Set current directory to be the working directory
cd(@__DIR__)

"""
    RIDC2(f::Function, u0::Vector{Float64}, T::Float64, N::Int64, M::Int64, K::Int64) -> (t::Vector{Float64}, u::Vector{Float64})

Calculates an approximate solution to the ODE: u'(t) = f(u, t), u(0) = u0, t∈[0, T]
via the revisionist integral deferred correction method (RIDC) with order 2 prediction
and corrections. The M correction steps give this method a theoretical order of accuracy
of 2(M+1). In practice the actual accuracy of the method is highly dependent on the 
parametes used and the nature of the ODE to be solved. 

# Inputs

f: RHS to the ODE: u'(t) = f(u, t)\n
u0: Initial condition vector\n
T: Maximum length of the time domain\n
N: Total number of time steps\n
M: Number of corrections\n
K: Number of nodes in each time interval (K >> M and K divides N)

# Outputs

t: Vector of the discretised time values\n
u: Matrix of approximated solution values at the discretised time values

# Examples
"""
function RIDC2(
    f::Function, 
    u0::Vector{Float64}, 
    T::Float64,
    N::Int64,
    M::Int64,
    K::Int64)

    # Number of time intervals
    J = Int(N / K)
    # Time step
    h = T / N
    # d is the number of equations in the system of ODEs
    d = length(u0)

    # Get quadrature weights matrix
    # Interpolating polynomial used is of degree 2*M+1 across all corrections
    interp_nodes = range(0, (2*M+1)*h, length = 2*M+2)
    quad_weights = zeros(2*M+1, 2*M+2)
    for i in 1:2*M+1
        for j in 1:2*M+2
            pol_coef = zeros(2*M+2)
            pol_coef[1] = 1
            for k in union(1:j-1, j+1:2*M+2)
                pol_coef = (
                    pol_coef * -interp_nodes[k] / (interp_nodes[j] - interp_nodes[k])
                    +
                    circshift(pol_coef, 1) / (interp_nodes[j] - interp_nodes[k])
                )
            end

            quad_weights[i, j] = (
                sum(pol_coef .* interp_nodes[i+1] .^ collect(1:2*M+2) ./ collect(1:2*M+2))
                -
                sum(pol_coef .* interp_nodes[i] .^ collect(1:2*M+2) ./ collect(1:2*M+2))
            )
        end
    end

    # Define order 2 prediction function (Heun's Method)
    function predict(
        u_pred0::Vector{Float64}, 
        t0::Float64)

        t1 = t0 + h
        du__dt_approx = f(u_pred0, t0)
        u_pred1_approx = u_pred0 + h*du__dt_approx
        u_pred1 = u_pred0 + 0.5*h*(du__dt_approx + f(u_pred1_approx, t1))
        return u_pred1
    end

    # Define order 2 correction function
    function correct(
        u_corr0::Vector{Float64}, 
        u_prev_lvl::Matrix{Float64}, 
        quad_nodes::Vector{Float64}, 
        offset::Int64)

        t0 = quad_nodes[offset]
        t1 = quad_nodes[offset+1]
        # Get derivative values of the previous correction level values
        f_prev_lvl = zeros(2*(M+1), d)
        for i in 1:2*(M+1)
            f_prev_lvl[i, :] = f(u_prev_lvl[i, :], quad_nodes[i])
        end
        # Get integration values of the interpolated polynomial of f(u, t)
        f_int = [sum(quad_weights[offset, :] .* f_prev_lvl[:, i]) for i in 1:d]
        K1 = h*(f(u_corr0, t0) - f(u_prev_lvl[offset], t0))
        K2 = h*(f(u_corr0 + K1 + f_int, t1) - f(u_prev_lvl[offset+1], t1))
        u_corr1 = u_corr0 + 0.5*K1 + 0.5*K2 + f_int
        return u_corr1
    end

    # u_tensor holds all calculated values across all correction levels
    u_tensor = zeros((M+1, N+1, d))
    # Initial condition
    u_tensor[:, 1, :] = repeat(reshape(u0, 1, d), M+1)

    # Iterate over the time intervals
    for j in 1:J
        println("Time Interval $j")

        # Get the time steps in current time interval of domain
        interval_steps = range((j-1)*K+1, j*K)

        # ------------------------------------------------------------ # 
        # Prerequisite calculations before parallelisation can start
        
        # Loop over the various correction levels
        for m in 1:M+1
            # Initial 3M+2-m calculations for m'th correction level
            for i in 1:3*M+2
                time_step = interval_steps[i]
                u_val0 = u_tensor[m, time_step, :]
                if m == 1
                    t0 = (time_step-1)*h
                    u_tensor[1, time_step+1, :] = predict(u_val0, t0)
                else
                    offset = 2*M+1

                    # Get values from previous correction levels for interpolation
                    u_prev_lvl = zeros(2*M+2, d)
                    quad_nodes = zeros(2*M+2)

                    # Required nodes are different for first 2M+1 corrections
                    if i <= 2*M+1
                        offset = i
                        for (index, t) in enumerate(interval_steps[1:2*M+2] .- 1)
                            u_prev_lvl[index, :] = u_tensor[m-1, t+1, :]
                            quad_nodes[index] = t*h
                        end
                    else
                        for (index, t) in enumerate(interval_steps[i-2*M-1:i])
                            u_prev_lvl[index, :] = u_tensor[m-1, t+1, :]
                            quad_nodes[index] = t*h
                        end
                    end


                    u_tensor[m, time_step+1, :] = (
                        correct(u_val0, u_prev_lvl, quad_nodes, offset)
                    )
                end
            end
        end

        # ------------------------------------------------------------ # 
        # Parallelisation should start here

        # Loop over the various correction levels
        for m in 1:M+1
            # Final K - (3*M+2-m) calculations for m'th correction level
            for i in 3*M+3-m:K
                time_step = interval_steps[i]
                u_val0 = u_tensor[m, time_step, :]
                if m == 1
                    t0 = (time_step-1)*h
                    u_tensor[1, time_step+1, :] = predict(u_val0, t0)
                else
                    offset = 2*M+1

                    # Get values from previous correction levels for interpolation
                    u_prev_lvl = zeros(2*M+2, d)
                    quad_nodes = zeros(2*M+2)

                    for (index, t) in enumerate(interval_steps[i-2*M-1:i])
                        u_prev_lvl[index, :] = u_tensor[m-1, t+1, :]
                        quad_nodes[index] = t*h
                    end

                    u_tensor[m, time_step+1, :] = (
                        correct(u_val0, u_prev_lvl, quad_nodes, offset)
                    )
                end
            end
        end

        # ------------------------------------------------------------ # 
        # Final correction value for the last value in this interval should
        # be dropped down to the prediction level

        u_tensor[:, interval_steps[end], :] = (
            repeat(u_tensor[end:end, interval_steps[end], :], M+1)
        )
    end

    t = range(0, T, length = N+1)
    return t, u_tensor[end, :, :]
end

function order_table(
    f::Function,
    U::Function,
    u0::Vector{Float64}, 
    T::Float64,
    M::Int64)

    error_list = []
    N_list = [40, 80, 120, 160, 200]
    K = 20
    for N in N_list
        t, u = RIDC2(f, u0, T, N, M, K)
        u_approx = u[end, :]
        u_true = U(T)

        push!(error_list, sqrt(sum((u_true - u_approx) .^2)))
    end

    order_list = ["NaN"; diff(log.(error_list)) ./ diff(log.(T ./ N_list))]

    order_table = DataFrame(
        N = N_list,
        Error = error_list, 
        Order = order_list
    )

    return order_table
end

U(t) = [exp(t)*(1-exp(-(t-1)^2))]
f(u, t) = [u[1]+2*(t-1)*(exp(t)-u[1])]
u0 = [1-exp(-1)]
T = 1.0
N = 1000
M = 1
K = 100

@time RIDC2(f, u0, T, N, M, K)

order_tab = order_table(f, U, u0, T, M)
display(order_tab)
CSV.write("RIDC2_OrderTable.csv", order_tab)