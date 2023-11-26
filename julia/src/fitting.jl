using LinearAlgebra
using SparseArrays
using Polynomials
using CVXOPT

# for logging
using Printf

using AutomaticDocstrings

"""
    get_gammas(Qp::Matrix{ComplexF64}, n_gamma::Int)

Obtain the gammas to construct the quadratic optimization problem

# Arguments:
- `Qp`: The phase-decorated eigen vectors obtained from the Takagi factorization
- `n_gamma`: The number of "gammas" that you request. A larger number means you want to use more exponential decays to describe the correlation function.
"""
function get_gammas(Qp::Matrix{ComplexF64}, n_gamma::Int)
    # if n==0, return an zero sized vector
    if n_gamma == 0
        return Vector()
    end
    solve_roots(v::Vector) = roots(Polynomial(v))

    # Note: The numpy roots API is reversed. That is p[0] * x^n + \dots + p[n] * x[0]
    # Whereas the julia `Polynomials.jl` roots are not reversed.
    # Hence, don't need to reverse Qp vectors
    # Note that although `PolynomialRoots.jl` are faster, the package "can give highly inaccurate results for polynomials of order above ~30". So don't use it for the prony fitting purpose, where the Hankel matrix can be large.
    gamma = solve_roots(Qp[:, n_gamma+1])
    argsort = sortperm(abs.(gamma))
    gamma_out = gamma[argsort][1:n_gamma]
    return Vector{ComplexF64}(gamma_out)
end

"""
    get_gamma_matrix(H::Hankel, gamma::Vector{Complex{Float64}})

Construct the gamma matrix for the quadratic optimization problem. Specifically, the gamma matrix is used to compute the quadratic term matrix.

# Arguments:
- `H`: The Hankel object
- `gamma`: The gamma values obtained from function `get_gammas`
"""
function get_gamma_matrix(H::Hankel, gamma::Vector{Complex{Float64}})
    n_col = 2 * H.dim + 1
    n_row = length(gamma)
    gamma_matrix = zeros(Float64, 2 * n_col, 2 * n_row)
    
    for i in 1:n_row
        for j in 1:n_col
            gamma_matrix[j, i] = real(gamma[i]^(j - 1))
            gamma_matrix[n_col + j, n_row + i] = real(gamma[i]^(j - 1))
            gamma_matrix[j, n_row + i] = -imag(gamma[i]^(j - 1))
            gamma_matrix[n_col + j, i] = imag(gamma[i]^(j - 1))
        end
    end
    
    return gamma_matrix
end

"""
    get_correlation_function_matrix(data::TimeDomainData)

Construct the correlation function vector. Which will the vector in the linear term of the Quadratic optimization problem. 

# Arguments:
- `data`: the `TimeDomainData` object representing a time domain correlation function. 
"""
function get_correlation_function_matrix(data::TimeDomainData)
    ct = data.correlation_function
    h_matrix = vcat(real(ct), imag(ct))
    return h_matrix
end

"""
    get_expn(data::TimeDomainData, t::Vector{ComplexF64})

Get the exponents from the t computed from the Takagi factorization Qp's

# Arguments:
- `data`: the `TimeDomainData` object representing a time domain correlation function. 
- `t`: the `t` vector
"""
function get_expn(data::TimeDomainData, t::Vector{ComplexF64})
    return -t / data.tf
end

"""
    get_freq_matrix(H::Hankel, data::TimeDomainData, t::Vector{ComplexF64})

This function computes the linear constraint matrix in the quadratic optimization problem, particularly, Ax <= 0.

# Arguments:
- `H`: the Hankel object
- `data`: the `TimeDomainData` object 
- `t`: the `t` vector t computed from the Takagi factorization Qp's
"""
function get_freq_matrix(H::Hankel, data::TimeDomainData, t::Vector{ComplexF64})
    n_col = 2 * H.dim + 1
    n_row = length(t)
    
    hi_freq_left = range(-10000, 10, div(n_col, 2))
    lo_freq = range(-10, 10, n_col + 1)
    hi_freq_right = range(10, 10000, div(n_col, 2))
    freq_d = vcat(hi_freq_left, lo_freq, hi_freq_right)
    
    expn = get_expn(data, t)
    freq_m = zeros(Float64, 2 * n_col, 2 * n_row)
    for i in 1:n_row
        for j in 1:2 * n_col
            freq_m[j, i] = real(expn[i]) / (real(expn[i])^2 + (imag(expn[i]) - freq_d[j])^2)
            freq_m[j, n_row + i] = (imag(expn[i]) - freq_d[j]) / (real(expn[i])^2 + (imag(expn[i]) - freq_d[j])^2)
        end
    end
    
    return freq_m
end

"""
    get_gammas_and_t(H::Hankel, n_gamma_real::Int, n_gamma_imag::Int)

Compute the gammas and t from the Hankel object, precisely the Takagi factorization data.

# Arguments:
- `H`: the Hankel object
- `n_gamma_real`: the number of real gamma. The more you choose, the more terms in the exponential decomposition.
- `n_gamma_imag`: the number of imaginary gammas. The more you choose, the more terms in the exponential decomposition.
"""
function get_gammas_and_t(H::Hankel, n_gamma_real::Int, n_gamma_imag::Int)
    gamma_real = get_gammas(H.Qp_of_H_real, n_gamma_real)
    gamma_imag = get_gammas(H.Qp_of_H_imag, n_gamma_imag)
    
    t_real = 2.0 * H.dim * log.(gamma_real)
    t_imag = 2.0 * H.dim * log.(gamma_imag)
    
    gamma = Vector{ComplexF64}(vcat(gamma_real, gamma_imag))
    t = Vector{ComplexF64}(vcat(t_real, t_imag))
    
    return gamma, t
end

@doc raw"""
    QP(Q::AbstractMatrix, A::AbstractMatrix, q::AbstractVector)

Construct an specific quadratic problem that is used in the prony fitting program.
Specifically, this specific optimization problem
```math
\begin{equation}
    \begin{aligned}
    \min & \quad & \frac{1}{2} x^{T} Q x + q^{T}x, \\
    \text{constraint} & \quad & A x <= 0.
    \end{aligned}
\end{equation}
```

# Arguments:
- `Q`: The quadratic matrix ``Q`` in ``x^{T} Q x``
- `G`: The linear constrant matrix ``G`` in ``Gx ≤ 0``
- `q`: The linear term to be optimized: ``q`` in ``q^{T} x``
"""
function QP(Q::AbstractMatrix, G::AbstractMatrix, q::AbstractVector)
    # sparsify the quadratic form matrix and the linear constraint matrix
    QCSC = sparse(tril(Q))
    GCSC = sparse(G)

    # set up the upper bound for Gx
    h = zeros(size(G, 1))

    opts = Dict(
        "show_progress" => true, 
        "abstol" => 1e-24, 
        "reltol" => 1e-24, 
        "feastol" => 1e-24
       )
    sol = CVXOPT.qp(QCSC, q, GCSC, h, options=opts)

    return sol
end

"""
    optimize(C, d, A)

The optimization function in prony fitting

# Arguments:
- `C`: The C matrix, i.e., gamma matrix computed from the Takagi factorizations
- `d`: The d vector, i.e., the correlation function vector.
- `A`: The A matrix, i.e., the frequency matrix, for constraint in optimization.
"""
function optimize(C, d, G)
    Q = C' * C
    q = -(d' * C)'

    sol = QP(Q, G, q)

    n_gamma = div(size(Q, 1), 2)
    # once again, due to julia has different approch to Vector
    # The construction of omega_new from sol["x"] is different than that in python
    omega_new_tmp = reshape(sol["x"], n_gamma, 2)
    omega_new = omega_new_tmp[:, 1] + 1.0im * omega_new_tmp[:, 2]

    return omega_new
end

# The prony main program
"""
    prony_fitting(data::TimeDomainData, nmode_real::Int, nmode_imag::Int, tol::Float64 = 1.0e-8)

The main program for the prony fitting

# Arguments:
- `data`: The time domain data organized in TimeDomainData data struct
- `nmode_real`: The number of modes requested the user 
- `nmode_imag`: The number of imaginary modes requested by the user 
- `tol`: The tolerance for the Takagi factorization
"""
function prony_fitting(data::TimeDomainData, nmode_real::Int, nmode_imag::Int, tol::Float64 = 1e-8)
    @info "Staring the prony fitting program."
    @info "Building the Hankel matrix and running Takagi factorization..."
    time_H = @elapsed H = Hankel(data.correlation_function, data.n_Hankel, tol)
    @info "The Hankel process is done!"
    @info "$H"
    @info "Solving for the gamma values from the eigen values..."
    time_roots = @elapsed gamma, t = get_gammas_and_t(H, nmode_real, nmode_imag)
    @info """
        The gamma and t values are solved!
        - gamma: $gamma; 
        -t: $t.
    """
    @info "Preparing the matrices for Prony optimization..."
    C = get_gamma_matrix(H, gamma)
    d = get_correlation_function_matrix(data)
    G = -get_freq_matrix(H, data, t)

    @info "Start the minimization..."
    time_opt = @elapsed  omega_new = optimize(C, d, G)
    @info "Minimization done, now outputting decomposed correlation function."
    
    etal = copy(omega_new)
    etar = conj.(omega_new)
    etaa = abs.(omega_new)
    expn = get_expn(data, t)

    @info @sprintf "
    Time elapsed
    - Hankel matrix and Takagi factorization: %.4f seconds.
    - calculate the gamma and t by finding the polynomials roots: %.4f seconds.
    - The quadratic optimization: %.4f seconds.
    " time_H time_roots time_opt
    return expn, etal
end

"""
    main()

A simple test function to use the julia prony fitting program.
"""
function main()
    β = 1.0 / 10
    data = TimeDomainData(BO, bose_function, β; tf=100)
    
    nmode_real = 1
    nmode_imag = 2

    expn, etal = prony_fitting(data, nmode_real, nmode_imag)

    len_ = 10000
    spe_wid = 10
    ω = range(-spe_wid, spe_wid, len_)

    jw_exact = @. BO(ω) * bose_function(ω; β=β)
    jw_prony = real(get_spectral_function_from_exponentials(collect(ω), expn, etal))
    return ω, jw_exact, jw_prony
end
