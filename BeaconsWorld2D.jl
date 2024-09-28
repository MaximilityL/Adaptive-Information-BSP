#include("structs.jl")

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    Dmax::Int64
    λ::Float64
    rng::MersenneTwister
    a_space::Array{Float64, 2}
    beacons::Array{Float64, 2}
    obstacles::Array{Float64, 2}
    d::Float64
    rmin::Float64
    obsRadii::Float64
    goalRadii::Float64
    goal::Array{Float64, 1}
    rewardGoal::Float64
    rewardObs::Float64
    γ::Float64
    w1::Float64
end


function update_observation_cov(pomdp, x)
    mindist = Inf
    for i in 1:length(pomdp.beacons[:,1])
        distance = norm(x[1:2] - pomdp.beacons[i,:])  # Use only the position part of x
        if distance <= pomdp.d
            pomdp.Σv = Matrix(Diagonal([0.01^2, 0.01^2]))  # 2D covariance for position
            return pomdp.Σv
        elseif distance < mindist
            mindist = distance
        end
    end
    # if no beacon is near by, get noise meas.
    pomdp.Σv = Matrix(Diagonal([0.1*mindist, 0.1*mindist]))  # 2D covariance for position
    return pomdp.Σv
end

function dynamics(x::Array{Float64, 1}, a::Array{Float64, 1}, rng)
    global pomdp
    return SampleMotionModel(pomdp, a, x)
end

function pdfObservationModel(x_prev::Vector{Float64}, a::Vector{Float64}, x::Vector{Float64}, obs::Vector{Float64})
    global pomdp
    pomdp.Σv = update_observation_cov(pomdp, x)
    
    # Only compare the position (first two elements)
    Nv = MvNormal([0, 0], pomdp.Σv[1:2, 1:2])
    noise = obs - x[1:2]  # Use only the position part of the state
    
    return pdf(Nv, noise)
end

function PropagateBelief(b::FullNormal, pomdp::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    Σw = pomdp.Σw  # Σv isn't used in propagation as it's related to observation noise

    v = a[1]  # Linear velocity
    ω = a[2]  # Angular velocity
    θ = μb[3]  # Current orientation

    Δx = v * cos(θ)
    Δy = v * sin(θ)
    new_θ = θ + ω

    # Predict new mean (μp)
    μp = [μb[1] + Δx, μb[2] + Δy, new_θ]

    # Create expected delta vector from the action
    Δ = [Δx, Δy, ω]

    # Adjust the covariance propagation to include the influence of Σw
    A = [ Σb^(-0.5) zeros(3,3); -Σw^(-0.5) Σw^(-0.5) ]
    b_adjusted = [Σb^(-0.5) * μb; Σw^(-0.5) * Δ]  # Apply Δ instead of action directly

    # Predict mean
    μp_adjusted = inv(transpose(A) * A) * (transpose(A) * b_adjusted)
    Σp = inv(transpose(A) * A)

    # Extract the 3D (x, y, θ) portion for the updated belief
    Σp = Σp[4:6, 4:6]  # Covariance for updated state (x, y, θ)
    μp = μp_adjusted[4:6]  # Mean for the updated state (x, y, θ)

    return MvNormal(μp, Σp)
end


# Input: belief at k, b(x_k), action a_k and observation z_k+1
# Output: updated posterior gaussian belief b(x')~N(μb′, Σb′)
function PropagateUpdateBelief(b::FullNormal, pomdp::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    Σw, Σv = pomdp.Σw, pomdp.Σv  # Process and observation noise covariance

    # Extract action dynamics
    v = a[1]  # Linear velocity
    ω = a[2]  # Angular velocity
    θ = μb[3]  # Current orientation

    # Propagate dynamics
    Δx = v * cos(θ)
    Δy = v * sin(θ)
    new_θ = θ + ω

    # Predicted new mean from dynamics
    μp = [μb[1] + Δx, μb[2] + Δy, new_θ]

    # Adjust the covariance propagation with respect to the action dynamics
    Δ = [Δx, Δy, ω]  # Vector of expected changes due to the action
    # Assuming Σv is a 2x2 matrix
    Σv_adjusted = [Σv[1,1] Σv[1,2] 0;
    Σv[2,1] Σv[2,2] 0;
    0       0       1]
   # Construct matrix A for prediction and update steps
    A = [ Σb^(-0.5) zeros(3,3);    # State prediction part (3x3 covariance for state, no influence on observation)
    -Σw^(-0.5) Σw^(-0.5);     # Process noise (3x3) acting on state and action
    zeros(3,3) Σv_adjusted^(-0.5)]    # Observation noise (2x2) acting on the observation, zeros for the state part

    
    # b_adjusted now reflects both the action dynamics and observation
    b_adjusted = [Σb^(-0.5) * μb; Σw^(-0.5) * Δ; Σv_adjusted^(-0.5) * [o; 0]]

    # Predict step: compute the updated mean and covariance
    μp_adjusted = inv(transpose(A) * A) * (transpose(A) * b_adjusted)
    Σp = inv(transpose(A) * A)

    # Extract updated mean and covariance (state and orientation)
    μb′ = μp_adjusted[1:3]  # The first 3 elements correspond to the updated state (x, y, θ)
    Σb′ = Σp[1:3, 1:3]      # Extract the 3x3 covariance matrix for the state

    return MvNormal(μb′, Σb′)
end


function SampleMotionModel(pomdp::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    # a[1] is linear velocity, a[2] is angular velocity
    v = a[1]  # Linear velocity
    ω = a[2]  # Angular velocity

    # Extract current orientation (angle) from the state
    θ = x[3]  # x[3] is the orientation (angle in radians)

    # Update position based on linear velocity and current orientation
    Δx = v * cos(θ)  # Change in x-position
    Δy = v * sin(θ)  # Change in y-position

    # Update the orientation based on angular velocity
    new_θ = θ + ω

    # Add noise to the movement (optional)
    Nw = MvNormal([0, 0, 0], pomdp.Σw)  # Motion noise
    w = rand(pomdp.rng, Nw)

    # Return the new state: [new_x, new_y, new_orientation]
    return [x[1] + Δx + w[1], x[2] + Δy + w[2], new_θ]
end

function pdfMotionModel(pomdp::POMDPscenario, a::Vector{Float64}, x::Vector{Float64}, x_prev::Vector{Float64})
    if length(x) == 3 && length(x_prev) == 3
        # Handle 3D state (x, y, theta)
        # Expected position change due to velocity (a[1]) and angular velocity (a[2]) over time Δt
        Δx_expected = a[1] * cos(x_prev[3])   # Linear velocity affects x
        Δy_expected = a[1] * sin(x_prev[3])  # Linear velocity affects y
        Δθ_expected = a[2]   # Angular velocity affects orientation (theta)
        
        # Calculate the difference between the actual state change and expected change
        w = [(x[1] - x_prev[1]) - Δx_expected,
             (x[2] - x_prev[2]) - Δy_expected,
             (x[3] - x_prev[3]) - Δθ_expected]

        Nw = MvNormal([0, 0, 0], pomdp.Σw)  # 3D noise for motion
    elseif length(x) == 2 && length(x_prev) == 2
        # Handle 2D state (x, y) only
        Nw = MvNormal([0, 0], pomdp.Σw[1:2, 1:2])  # 2D covariance for position
        w = x - x_prev - a[1:2]  # Only take position from action
    else
        throw(DimensionMismatch("x and x_prev must both be 2D or 3D vectors"))
    end

    return pdf(Nw, w)
end

function GenerateObservationFromBeacons(pomdp::POMDPscenario, x::Array{Float64, 1}, fixed_cov::Bool)::Union{NamedTuple, Nothing}
    distances = zeros(length(pomdp.beacons[:,1]))
    for index in 1:length(pomdp.beacons[:,1])
        distances[index] = norm(x[1:2] - pomdp.beacons[index, :])  # Position only
    end
    index = argmin(distances)  # Nearest beacon

    pomdp.Σv = update_observation_cov(pomdp, x)
    Nv = MvNormal([0, 0], pomdp.Σv)  # Measurement noise
    v = rand(pomdp.rng, Nv)
    dX = x[1:2] - pomdp.beacons[index, :]

    # Optionally include orientation in observation
    obs = [dX[1] + v[1], dX[2] + v[2]]  # Including orientation (if relevant)
    return (obs=obs, index=index)
end


function SampleObservation(p::Planner, x_propagated::Array{Float64, 1})
    o = GenerateObservationFromBeacons(p.pomdp, x_propagated, false)
    
    # Make sure it returns only the position component if you're only dealing with 2D observations
    return o.obs + p.pomdp.beacons[o.index, :]
end

function likelihood(x::Vector{Float64}, o::Vector{Float64})
    return pdfObservationModel([0.0, 0.0], [0.0, 0.0], x[1:2], o)  # Compare only the position part
end

function initBelief()
    μ0 = [0.0, 0.0, 0.0]  # Initial position and orientation
    Σ0 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.1]  # Covariance for position and orientation
    return MvNormal(μ0, Σ0)
end
# initState() = [-0.5, -0.2]

function initState()
    return [-0.5, -0.2, 0.0]  # Initial position (x, y) and orientation (θ)
end

function reward(p::Planner, b::FullNormal, x::Vector{Float64})
    x_g, x_o = p.pomdp.goal, p.pomdp.obstacles[1,:]
    rewardObs, rewardGoal = 0, 0
    n = length(b.μ) # dimension of state vector
    if norm(x[1:2] - x_o, 2) < pomdp.obsRadii
        rewardObs = p.pomdp.rewardObs
    end
    if norm(x[1:2] - x_g, 2) < pomdp.goalRadii
        rewardGoal = p.pomdp.rewardGoal
    end
    return -(p.pomdp.w1 * norm(b.μ[1:2]-x_g,2)+p.pomdp.λ*0.5*log((2* π *exp(1))^n * det(b.Σ))) + rewardObs + rewardGoal
end


function exactReward(p, b, ba, a, ba_id)
    # for debug porpuses only, w/o obstacles
    r_belief, eta = 0, 0
    for bao_id in p.tree.nodes[ba_id].children
        p_z_x = p.tree.nodes[bao_id].likelihood
        (unnorm_r, normalizer) = rewardBelief(p, b, ba, a, p_z_x)
        r_belief += unnorm_r
        eta += normalizer
    end
    r_belief = r_belief / eta
    r = p.pomdp.w1 * rewardState(p, b) + p.pomdp.λ*r_belief
    return r
end


function oneStepSim(p::Planner, b::FullNormal, x_prev::Array{Float64, 1}, a::Array{Float64, 1})
    # create GT Trajectory, update horizon
    p.pomdp.Dmax -= 1

    # Print the selected action (v, ω)
    println("Selected action (v, ω): ", a[1], ", ", a[2])

    b_prop = PropagateBelief(b, p.pomdp, a)
    x = SampleMotionModel(p.pomdp, a, x_prev)

    # Print the current x, y, and θ (theta) values
    println("Current state (x, y, θ): ", x[1], ", ", x[2], ", ", x[3])

    o_rel = GenerateObservationFromBeacons(p.pomdp, x, false)
    if o_rel === nothing
        b_post = b_prop
    else
        o = o_rel[1] + p.pomdp.beacons[o_rel[2], :]
        # update Cov. according to distance from beacon
        update_observation_cov(p.pomdp, x)
        b_post = PropagateUpdateBelief(b_prop, p.pomdp, a, o)
        r = reward(p, b_post, x)
    end

    return b_post, r, x, o
end


function BeaconsWorld2D(rng)
    d = 1.0 
    rmin = 0.1
    linear_velocity_norm = 2.0
    angular_velocity_norm = deg2rad(90)

    # set beacons locations 
    beacons = [0.0 0.0; 
               #2.0 0.0; 
               4.0 0.0;
               #6.0 0.0;
               8.0 0.0;
               #10.0 0.0;
               10.0 2.0;
               #10.0 4.0;
               10.0 6.0;
               #10.0 8.0;
               10.0 10.0;]
    obstacles = [10.0 * rand(rng,1) 5.0;
                 10.0 * rand(rng,1) 3.0;
                 10.0 * rand(rng,1) 9.0;]
    goal = [10, 10]
    a_space = [linear_velocity_norm  0.0;  # Move forward with velocity linear_velocity_norm
          -linear_velocity_norm  0.0;  # Move backward with velocity -linear_velocity_norm
           0.0  angular_velocity_norm;  # Turn right with angular velocity angular_velocity_norm
           0.0 -angular_velocity_norm;  # Turn left with angular velocity -angular_velocity_norm
           linear_velocity_norm  angular_velocity_norm;  # Move forward while turning right
           linear_velocity_norm -angular_velocity_norm;  # Move forward while turning left
          -linear_velocity_norm  angular_velocity_norm;  # Move backward while turning right
          -linear_velocity_norm -angular_velocity_norm;
          0.0 0.0]  # Don't Change Nothing
    pomdp = POMDPscenario(F = [1.0 0.0 0.0;
     0.0 1.0 0.0;
     0.0 0.0 1.0],
    Σw = 0.1 * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.05],
    Σv = 0.01 * [1.0 0.0; 0.0 1.0], 
                        γ=0.99,
                        Dmax = 25,
                        λ = 1,
                        obsRadii = 1.5,
                        goalRadii = 1.,
                        goal = goal,
                        rewardGoal = 20,
                        rewardObs = -10,
                        rng = rng, a_space=a_space, beacons=beacons, 
                        obstacles=obstacles, d=d, rmin=rmin,
                        w1 = 1.0) 
    global pomdp

    return pomdp
end