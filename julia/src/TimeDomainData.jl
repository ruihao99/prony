using LinearAlgebra
using FFTW
using Plots

"""
    TimeDomainData

DOCSTRING

# Fields:
- `n_sample::Int`: number of frequency domain samples
- `max_freq_in_pi::Int`: the maximum frequency for the frequency domain sampling of the spectral function. The unit is π.
- `β::Float64`: the inverse temperature
- `tf::Float64`: the final time 
- `n_Hankel::Int`: the dimension of the Hankel Matrix
- `ω::Vector{Float64}`: the sampled frequency data
- `spectral_function::Vector{ComplexF64}`: the spectral function
- `time::Vector{Float64}`: the sampled time 
- `correlation_function::Vector{ComplexF64}`: the sampled time domain correlation function
"""
struct TimeDomainData
    n_sample::Int
    max_freq_in_pi::Int
    β::Float64
    tf::Float64
    n_Hankel::Int
    ω::Vector{Float64}
    spectral_function::Vector{ComplexF64}
    time::Vector{Float64}
    correlation_function::Vector{ComplexF64}
end

"""
    TimeDomainData(spectral_function::Function, bath_statistic_function::Function, β::Float64, μ::Float64=0.0, tf::Integer = 200, n_Hankel::Integer = 2000, n_sample::Integer = 1000000, max_freq_in_pi::Integer = 3000)

DOCSTRING

# Arguments:
- `spectral_function`: The spectral function
- `bath_statistic_function`: The bath statistic function, i.e., the fermi function or the bose function
- `β`: the inverse temperature
- `μ`: the chemical potential
- `tf`: the final time for the time domain samples
### - `n_Hankel`: the dimension for the Hankel matrix
- `n_sample`: the number of frequency domain samples
- `max_freq_in_pi`: the maximum frequency to be sampled, in the unit of π
"""
function TimeDomainData(
    spectral_function::Function, 
    bath_statistic_function::Function,
    β::Float64, 
    μ::Float64=0.0,
    tf::Integer=200, 
    # n_Hankel::Integer=2000, 
    n_sample::Integer=1000000, 
    max_freq_in_pi::Integer=3000
)

    n_Hankel = tf * 10
    ω = range(0, max_freq_in_pi * π, n_sample+1)[1:end-1]
    Jω = @. spectral_function(ω)

    f(ω) = bath_statistic_function(ω; β=β, μ=μ)

    dω = ω[2] - ω[1]
    cω_pos = @. Jω * f(+ω)
    cω_neg = @. Jω * f(-ω) 
    cω_pos[1] = cω_pos[2] / 2
    cω_neg[1] = cω_neg[2] / 2

    ct = dω / π * (FFTW.fft(cω_pos) .- FFTW.ifft(cω_neg) .* length(cω_neg))

    # note here FFTW has different fftfreq in comparison to numpy
    # 1.0/dω in julia, and dω in python
    time = 2.0 * π * FFTW.fftfreq(length(ct), 1.0/dω)

    n_rate = floor(Integer, max_freq_in_pi * tf / (4 * n_Hankel))
    non_negative_time_mask = (time .<= tf) .& (time .>= 0)

    time = time[non_negative_time_mask][1:n_rate:end]
    ct = ct[non_negative_time_mask][1:n_rate:end]

    return TimeDomainData(n_sample, max_freq_in_pi, β, tf, n_Hankel, ω, Jω, time, ct)
end

"""
    plot_correlation_function(t, ct)

DOCSTRING

# Arguments:
- `t`: The evenly sampled time
- `ct`: The time domain correlation function 
"""
function plot_correlation_function(t, ct)
    plt = plot()
    plot!(t, imag.(ct), label="Im")
    plot!(t, real.(ct), label="Re", xlabel="Time", ylabel="C(t)", xlims=(-0.1, 50))
end

"""
    plot_time_domain_data(data::TimeDomainData)

DOCSTRING

# Arguments:
- `data`: The time TimeDomainData struct to organize data
"""
function plot_time_domain_data(data::TimeDomainData)
    plot_correlation_function(data.time, data.correlation_function)
end

# Define your bose_function here, assuming it's similar to the Python implementation

# Example usage:
# spectral_function = w -> your_spectral_function_here(w)
# data = TimeDomainData(spectral_function, beta_value)
# plot(data)

