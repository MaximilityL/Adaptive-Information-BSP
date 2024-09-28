
# propegate belief
function ParticleBeliefMDP(b::ParticleCollection, pomdp::POMDPscenario, a::Array{Float64, 1}, pf, o::Array{Float64, 1})
    bp = predict(pf, b, a, pomdp.rng)
    b_post = update(pf, b, a, o)
    return b_post
end

function bores_entropy(pomdp::POMDPscenario, ba::ParticleCollection, likelihood::Array{Float64}, 
                       a::Array{Float64}, b::ParticleCollection)

    N = length(particles(ba))  # Number of particles in ba
    M = length(particles(b))   # Number of particles in b


    summ = sum(likelihood)
    
    # Normalize likelihood if the sum is not zero, otherwise keep it unchanged
    bao_weights = (summ != 0.0) ? likelihood ./ summ : likelihood

    normalizer, nominator = 0.0, 0.0

    for i in 1:N
        x = particles(ba)[i][1:2]  # Use only the first two elements (x, y)
        pdf_prop = 0.0  # Initialize pdf_prop inside the loop
        
        for j in 1:M
            x_prev = particles(b)[j][1:2]  # Use only the first two elements (x, y)
            
            # Now we are only handling 2D state vectors
            pdf_prop += pdfMotionModel(pomdp, a[1:2], x, x_prev) / M
        end

        # Only compute log if likelihood[i] is positive (avoid negative log issues)
        if likelihood[i] > eps()  # Using the default eps value
            nominator += log(likelihood[i] * pdf_prop) * bao_weights[i]
            normalizer += likelihood[i] / N
        end
    end

    # Check for numerical stability before logging
    normalizer = normalizer > 0.0 ? log(normalizer) : 0.0

    return normalizer - nominator
end


function expected_entropy(ba::ParticleCollection, likelihood::Array{Float64}, a::Vector{Float64}, b::ParticleCollection)
    N = length(particles(ba))
    M = length(particles(b))
    pdf_prop = 0
    bao_weights = likelihood ./ N  # Divide by N to get posterior weight
    denominator, nominator = 0, 0

    for i in 1:N
        x = particles(ba)[i]
        for j in 1:M
            x_prev = particles(b)[j]
            # Ensure consistent dimensions when calling pdfMotionModel
            pdf_prop += pdfMotionModel(pomdp, a, x[1:2], x_prev[1:2]) / M  # Use only the position part
        end
        if likelihood[i] > eps(10^-100)  # avoid rounding errors within log
            nominator += log(likelihood[i] * pdf_prop) * bao_weights[i]
            denominator += likelihood[i] / N
        end
    end
    denominator = sum(bao_weights) * log(denominator)
    unnormalized_entropy = denominator - nominator
    normalizer = sum(bao_weights)
    return -unnormalized_entropy, normalizer
end

function posterior(p, b, a, bp, o)
    weights = reweight(p.solver.PF, b, a, particles(bp), o)
    bw = WeightedParticleBelief(particles(bp), weights, sum(weights), nothing)
    posterior = resample(LowVarianceResampler(
        length(particles(b))), bw, p.solver.PF.predict_model.f,
        p.solver.PF.reweight_model.g, 
        b, a, o, p.pomdp.rng) 
    return posterior
end

function posterior(p, b, a)
    bp = predict(p.solver.PF, b, a, p.pomdp.rng)
    x = rand(p.pomdp.rng, bp)
    o = SampleObservation(p, x)
    weights = reweight(p.solver.PF, b, a, bp, o)
    bw = WeightedParticleBelief(bp, weights, sum(weights), nothing)
    posterior = resample(LowVarianceResampler(
        length(particles(b))), bw, p.solver.PF.predict_model.f,
        p.solver.PF.reweight_model.g, 
        b, a, o, p.pomdp.rng)
    return (posterior, bp, o)
end




