export 
    bose_function,
    fermi_function,
    BO


"""
    bose_function(E; β::Float64, μ::Float64 = 0.0)

DOCSTRING

# Arguments:
- `E`: the energy
- `β`: the inverse temperature
- `μ`: the chemical potential
"""
bose_function(E; β::Float64, μ::Float64=0.0) = 1.0 / (1.0 - exp(-(E-μ) * β))

"""
    fermi_function(E; β::Float64, μ::Float64)

DOCSTRING

# Arguments:
- `E`: the energy 
- `β`: the inverse temperature 
- `μ`: the chemical potential, i.e., inverse temperature.
"""
fermi_function(E; β::Float64, μ::Float64=0.0) = 1.0 / (1.0 + exp((E - μ) * β))

"""
    get_correlation_spectra_from_exponentials(ω, exponent, η)

DOCSTRING

# Arguments:
- `ω`: The frequency sample
- `exponent`: The exponents
- `η`: the coeffcient
"""
function get_correlation_spectra_from_exponentials(ω::Vector, exponents::Vector, η::Vector)
    if length(exponents) != length(η)
        throw(ArgumentError("Exponents and coefficients must have identical dimensions. exp: $(length(exponents)) elements, η: $(length(η)) elements."))
    end

    J_ω = zero(ω)
    for i = eachindex(η)
        J_ω += @. η[i] / (exponents[i] - 1.0im * ω)
    end
    return J_ω
end

"""
    BO(ω, λ = 1.0, ζ = 1.0, ΩB = 1.0)

DOCSTRING

# Arguments:
- `ω`: the frequency
- `λ`: the interaction strength
- `ζ`: the secondary bath reorganalization energy
- `ΩB`: the frequency of the solvation mode 
"""
function BO(ω, λ::Float64=1.0, ζ::Float64=1.0, ΩB::Float64=1.0)
    return 2.0 * λ * ω * ΩB^2 / ((ω * ζ)^2 + (ω^2 - ΩB^2)^2)
end
