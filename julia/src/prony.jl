module prony

export 
    prony_fitting,
    TimeDomainData,
    get_correlation_spectra_from_exponentials


include("spectral.jl")
include("TimeDomainData.jl")
include("Hankel.jl")
include("fitting.jl")

end
