# prony

Spectral function decomposition using time domain prony fitting but in `julia` language.

Somewhat faster than the `python` version.

## quick usage

This code is not intendend to be published as an onlne `julia` package. Nonetheless, you can still easily use this package with ease.

First, go to the julia package directory

```bash
cd /path/to/prony/julia
julia
```

Then, simply add this local package to your Pkg manager
```julia
]
dev .
```

This is a quick example of how you use this `julia` package
```julia
using prony

# construct your time domain data from an analytical spectral function
# Here I use the BO funciton defined in `spectral.jl`, where the parameters `λ`, `ζ`, and `ΩB` are default value
# you can easily define an partial function that has different parameter sets
# BO_not_default = ω -> BO(ω; λ=your_λ, ζ=your_ζ, ΩB=your_ΩB)
β = 1.0 / 10
data = TimeDomainData(BO, bose_function, β; tf=100)

# choose numbers of exponential decays you want to use to decompose the spectral function
# real for the the real part of the correlation function (TimeDomainData); imag for imaginary part of TimeDomainData.
nmode_real = 1
nmode_imag = 2

# the fitting process, returns the exponents and the corrsponding η coefficients
expn, etal = prony_fitting(data, nmode_real, nmode_imag)

# comparison between the exact correlation spectra and exponential decompsed one
len_ = 10000
spe_wid = 10
ω = range(-spe_wid, spe_wid, len_)

jw_exact = @. BO(ω) * bose_function(ω; β=β)
jw_prony = real(get_correlation_spectra_from_exponentials(collect(ω), expn, etal))

devi = jw_exact - jw_prony
println("The maximum difference is $(maximum(abs.(devi)))")

# plot the difference
using Plots

plt = plot()
plot!(plt, ω, jw_exact, label="exact")
plot!(plt, ω, jw_prony, label="prony")

plot(ω, devi, label="deviation")
```
