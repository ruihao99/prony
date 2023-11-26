using LinearAlgebra

"""
    Hankel: simple structure to hold the Hankel matrix data

Data structure that curates the Hankel square matrices and the corresponding data for Takagi factorizations.

# Fields:
- `dim::Int`: the dimension of the Hankel matrix
- `H_real::Matrix{Float64}`: The real component of the Hankel matrix
- `H_imag::Matrix{Float64}`: The imaginary component of the Hankel matrix
- `abs_evals_of_H_real::Vector{Float64}`: The absolute value of eigen values for H_real, sorted.
- `Qp_of_H_real::Matrix{ComplexF64}`: The corresponding eigen vectors for H_real
- `error_real::Float64`: The error of the Takagi factorization for H_real
- `abs_evals_of_H_imag::Vector{Float64}`: The absolute value of eigen values for H_imag, sorted.
- `Qp_of_H_imag::Matrix{ComplexF64}`: The corresponding eigen vectors for H_imag
- `error_imag::Float64`: The error of the Takagi factorization for H_imag
"""
struct Hankel
    dim::Int
    H_real::Matrix{Float64}
    H_imag::Matrix{Float64}
    abs_evals_of_H_real::Vector{Float64}
    Qp_of_H_real::Matrix{ComplexF64}
    error_real::Float64
    abs_evals_of_H_imag::Vector{Float64}
    Qp_of_H_imag::Matrix{ComplexF64}
    error_imag::Float64
end

"""
    Hankel(correlation_function::Vector{Float64}, n_Hankel::Int, tol::Float64 = 1.0e-8)

The constructor for the Hankel data. The constructor will 
    - build the Hankel square matrix
    - factorize the Hankel square matrix
    - calculate the error of the factorization

# Arguments:
- `correlation_function`: Complexed valued correlation function in time domain 
- `n_Hankel`: The dimension of the Hankel square matrix
- `tol`: The tolerance for the Takagi factorization.
"""
function Hankel(correlation_function::Vector{ComplexF64}, n_Hankel, tol::Float64=1e-8)
    if !_is_valid_sample(correlation_function, n_Hankel)
        throw(ArgumentError("The dimension for Hankel matrix is $n_Hankel, which is too large for a correlation function with $(length(correlation_function)) sampled points."))
    end

    H_real, H_imag = get_Hankel(correlation_function, n_Hankel)
    abs_evals_of_H_real, Qp_of_H_real, error_real = factorize_Hankel(H_real, tol)
    abs_evals_of_H_imag, Qp_of_H_imag, error_imag = factorize_Hankel(H_imag, tol)

    return Hankel(n_Hankel, H_real, H_imag, abs_evals_of_H_real, Qp_of_H_real, error_real, abs_evals_of_H_imag, Qp_of_H_imag, error_imag)
end

"""
    get_Hankel(correlation_function::Vector{Float64}, n_Hankel::Int)

Construct the real and imaginary Hankel square matrix corresponding to the real and imaginary part of the correlation function

# Arguments:
- `correlation_function`: The time domain correlation function data 
- `n_Hankel`: The dimension of the Hankel matrix
"""
function get_Hankel(correlation_function::Vector{ComplexF64}, n_Hankel::Int)
    H_real = zeros(n_Hankel, n_Hankel)
    H_imag = zeros(n_Hankel, n_Hankel)

    for i in 1:n_Hankel
        H_real[i, :] .= real(correlation_function[i:n_Hankel+i-1])
        H_imag[i, :] .= imag(correlation_function[i:n_Hankel+i-1])
    end
  
    return H_real, H_imag
end

"""
    factorize_Hankel(M_hankel::Matrix{Float64}, tol::Float64)

Factorize a Hankel matrix using the Takagi factorization. The factorization error is calculated.

# Arguments:
- `M_hankel`: The Hankel matrix
- `tol`: The tolerance for the Takagi factorization.
"""
function factorize_Hankel(M_hankel::Matrix{Float64}, tol::Float64)

    abs_evals, Qp = Takagi_factorization(M_hankel)

    error = get_error_of_Takagi_factorization(M_hankel, abs_evals, Qp)
    if error >= tol
        throw(ArgumentError("The Takagi factorization of the matrix has error $error > the tolerance value $tol."))
    end

    return abs_evals, Qp, error
end

"""
    _is_valid_sample(correlation_function::Vector, nsample::Int)

A flag function that indicates whether the time domain correlation function has enough data to construct the n_Hankel-sized square Hankel matrices

# Arguments:
- `correlation_function`: The time domain correlation function data 
- `n_Hankel`: The dimension of the Hankel matrix
"""
function _is_valid_sample(correlation_function::Vector, n_Hankel::Int)
    ndata = length(correlation_function)
    return 2 * n_Hankel <= ndata
end

"""
    Takagi_factorization(H::Matrix{Float64})

The Takagi factorization of a square matrix. The output eigen values and eigen vectors will be used to extract out the most significant feature of the time domain correlation function.

# Arguments:
- `H`: The Hankel matrix
"""
function Takagi_factorization(H::Matrix{Float64})
    
    # Check if H is real and symmetric 
    if !isreal(H) || !issymmetric(H)
        throw(ArgumentError("The Hankel square matrices are required to be real and symmetric tridiagonal."))
    end

    evals, evecs = eigen(H)

    phase_mat = Diagonal([exp(-1.0im * angle(sing_v_r) / 2.0) for sing_v_r in evals])
    Qp = evecs * phase_mat

    abs_evals = abs.(evals)
    argsort = sortperm(abs_evals, rev=true)

    abs_evals = abs_evals[argsort]
    Qp = Qp[:, argsort]

    return abs_evals, Qp
end

"""
    get_error_of_Takagi_factorization(H::Matrix{Float64}, abs_evals::Vector{Float64}, Q::Matrix{ComplexF64})

Helper function that evaluates the error of the Takagi factorization.

# Arguments:
- `H`: The Hankel matrix
- `abs_evals`: The absolute eigen values from the Takagi factorization, sorted.
- `Q`: The corresponding eigen vectors of the absolute eigen values 
"""
function get_error_of_Takagi_factorization(H::Matrix{Float64}, abs_evals::Vector{Float64}, Q::Matrix{ComplexF64})
    H_from_Takagi = Q * Diagonal(abs_evals) * transpose(Q)
    return sum(abs.(H - H_from_Takagi))
end

"""
    Base.show(io::IO, h::Hankel)

Help display the important informations for the Hankel structure 
"""
function Base.show(io::IO, h::Hankel)
    print(io, "<Hankel dim=$(h.dim), with Takagi factorization error (real: $(h.error_real), imag: $(h.error_imag))>")
end
