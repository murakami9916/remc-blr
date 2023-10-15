using Random
using Statistics
using LinearAlgebra
using Distributions
using Plots
using ProgressBars
using DataFrames
using CSV
using StatsBase
using JLD2
using FileIO

### 構造体
# -----------------------------------------
struct Data
    X::Matrix{Float64}
    y::Vector{Float64}
    λ₀::Float64
    λ_vector::Vector{Float64}
    labels::Vector{String}
    Data(X, y, λ₀, λ_vector, labels) = new(X, y, λ₀,λ_vector,  labels)
end

mutable struct Parameters
    g::Vector{Int64}
    Parameters(g) = new(g)
end

mutable struct Replica
    Θ::Parameters
    β::Float64
    Eᵤ::Float64
    χ::Int64
    stack::Dict{String,Float64}
    function Replica(Θ, β, Eᵤ)
        χ = 0
        stack = Dict{String,Float64}()
        new(Θ, β, Eᵤ, χ, stack)
    end 
end

mutable struct Result
    g::Matrix{Int64}
    E::Vector{Float64}
    β::Float64
    χ::Int64
end
# -----------------------------------------

@inline function calc_posterior_distribution(
                X::Matrix{Float64}, y::Vector{Float64},
                λ::Float64, Λ₀::Matrix{Float64})
    Λₙ::Matrix{Float64} = Λ₀ + λ*X'*X
    μₙ::Vector{Float64} = ( (λ * y' * X) / Λₙ )'
    return μₙ, Λₙ
end

@inline function calc_model_evidence(
                y::Vector{Float64}, λ::Float64, μₙ::Vector{Float64},
                Λₙ::Matrix{Float64}, Λ₀::Matrix{Float64})
    N::Int64 = length(y)
    evidence::Float64 = 0.5 * ( 
        λ*sum(y.^2.0) + N*( -log(λ) + log(2π) )
        - μₙ'*Λₙ*μₙ + logdet(Λₙ) - logdet(Λ₀)
    )
    return evidence
end

# 台形近似を用いた1次元数値積分
@inline function trapezoidal_integration(
                x_values::Vector{Float64}, y_values::Vector{Float64})
    n::Int64 = length(x_values)
    integral::Float64 = 0.0
    for i in 1:n-1
        h::Float64 = abs(x_values[i+1] - x_values[i])
        integral += (y_values[i] + y_values[i+1]) * h / 2.0
    end
    return integral
end

@inline function calc_marginal_likelihood_over_lambda(
                X::Matrix{Float64}, y::Vector{Float64},
                Λ₀::Matrix{Float64}, λ_vector::Vector{Float64})
    M::Int64 = length(λ_vector)
    ϕ = Array{Float64}(undef, M)
    for m in 1:M
        λ::Float64 = λ_vector[m]
        μₙ, Λₙ = calc_posterior_distribution(X, y, λ, Λ₀)
        evidence::Float64  = calc_model_evidence(y, λ, μₙ, Λₙ, Λ₀)
        ϕ[m] = evidence
    end
    return ϕ
end

@inline function integrate_over_lambda(
                λ_vector::Vector{Float64}, ϕ::Vector{Float64})
    ϕ_max = maximum(-ϕ)
    F::Float64 = 0
    F = trapezoidal_integration( λ_vector, exp.(-ϕ.-ϕ_max) )
    F = - (log(F)+ϕ_max)
    return F
end

@inline function calculate_marginal_likelihood(data::Data, parameters::Parameters)
    # 事前分布の精度パラメータ
    L::Int64 = sum(parameters.g)
    Λ₀::Matrix{Float64} = data.λ₀ * Matrix(I, L, L)

    Xₛ = data.X[:, Bool.(parameters.g)]
    y = data.y; λ_vector = data.λ_vector
    ϕ::Vector{Float64} = calc_marginal_likelihood_over_lambda(Xₛ, y, Λ₀, λ_vector)
    FE::Float64 = integrate_over_lambda(λ_vector, ϕ)

    return FE
end

function metropolis_indicator(data::Data, replica::Replica)
    L::Int64 = length(replica.Θ.g)
    β::Float64 = replica.β
    for i in 1:L
        Θₜ = deepcopy(replica.Θ)
        Θₜ.g[i] = ( ~Θₜ.g[i] + 2 )

        pattern::String = join(string.(Θₜ.g))
        if haskey(replica.stack, pattern)
            tmp_Eᵤ = replica.stack[pattern]
        else
            tmp_Eᵤ = calculate_marginal_likelihood(data, Θₜ)
            replica.stack[pattern] = tmp_Eᵤ
        end
    
        if rand() <= exp( - β * (tmp_Eᵤ - replica.Eᵤ) )
            replica.Θ = deepcopy(Θₜ)
            replica.Eᵤ = deepcopy(tmp_Eᵤ)
        end
    end
    return replica
end

function raplica_exchange(replica_i::Replica, replica_j::Replica)
    Δβ::Float64 = ( replica_i.β - replica_j.β )
    ΔEᵤ::Float64 = ( replica_i.Eᵤ - replica_j.Eᵤ )
    if rand() < exp( Δβ*ΔEᵤ )
        replica_j.Θ, replica_i.Θ = deepcopy(replica_i.Θ), deepcopy(replica_j.Θ)
        replica_j.Eᵤ, replica_i.Eᵤ = deepcopy(replica_i.Eᵤ), deepcopy(replica_j.Eᵤ)
        replica_i.χ = 1; replica_j.χ = 1;
    else
        replica_i.χ = 0; replica_j.χ = 0;
    end
    return replica_i, replica_j
end

function remc_step(mcmc_step::Int64, data::Data, replicas::Vector{Replica})
    # メトロポリスステップ
    num_replicas::Int64 = length(replicas)
    for r in 1:num_replicas
        replicas[r] = metropolis_indicator(data, replicas[r])
    end
    
    # レプリカ交換
    start::Int64 = mcmc_step%2+1
    for r in start:2:num_replicas-1
        replicas[r], replicas[r+1] = raplica_exchange(replicas[r], replicas[r+1])
    end
    
    return replicas
end

function replica_exchange_mcmc(data::Data, replicas::Vector{Replica}, num_steps::Int64)
    # バーンイン
    # --------------------------------------------------------------
    println("[burn-in]")
    num_replicas::Int64 = length(replicas)
    for mcmc_step in ProgressBar(1:num_steps)
        replicas = remc_step(mcmc_step, data, replicas)
    end
    # --------------------------------------------------------------
    

    # 結果変数の初期化
    # --------------------------------------------------------------
    results = Array{Result}(undef, num_replicas)
    for r in 1:num_replicas
        g = Array{Int64}(undef, num_steps, length(replicas[1].Θ.g))
        β = replicas[r].β; E = Array{Float64}(undef, num_steps); χ = 0
        results[r] = Result(g, E, β, χ)
    end
    # --------------------------------------------------------------
    

    # 本番
    # --------------------------------------------------------------
    println("[Production sampling]")
    for mcmc_step in ProgressBar(1:num_steps)
        replicas = remc_step(mcmc_step, data, replicas)
        for r in 1:num_replicas
            results[r].g[mcmc_step, :] = replicas[r].Θ.g
            results[r].E[mcmc_step] = replicas[r].Eᵤ
            results[r].χ += replicas[r].χ
            replicas[r].χ = 0
        end
    end
    # --------------------------------------------------------------

    return results
end

function calc_DoS(B::Vector{Float64}, results::Vector{Result})
    # マルチヒストグラムによるDoSの計算
    T = length(B)
    M = size(results[T].g)[2]
    num_states = 2^M
    E_all = Array{Float64}(undef, num_steps, T)
    for r in 1:T
        E_all[:, r] = results[r].E
    end
    all_neg_log_likelihood = vec(E_all)
    n = size(E_all)[1] 
    hist = fit(Histogram, all_neg_log_likelihood, nbins=100)

    edges = Vector(hist.edges[1])
    E = (edges[1:end-1] + edges[2:end]) ./ 2.0
    H = hist.weights
    tolerance = 1e-6
    max_iter = 1000

    Z = ones(T)
    D = ones(length(H))
    for i in 1:max_iter
        prev_Z = deepcopy(Z)

        D = vec( H ./ ( exp.( - E*B' ) * ( n ./ Z ) ) )
        Z = vec( D' * exp.( - E*B' ) )
        
        err = mean(abs.(Z .- prev_Z))
        if err < tolerance
            println("i=$i, err=$err")
            break
        end
    end

    plt = bar( E[D.!=0], num_states .* ( D[D.!=0] / sum(D[D.!=0]) ), legend=false, 
            xlabel="Negative marginal log-likelihood", ylabel = "Density of states",
            dpi=500, color="gray")
    savefig(plt, "DoS_linear.png")


    plt = bar( E[D.!=0], num_states .* ( D[D.!=0] / sum(D[D.!=0]) ), legend=false, 
            xlabel="Negative marginal log-likelihood", ylabel = "Density of states",yscale=:log10,
            dpi=500, color="gray")
    savefig(plt, "DoS_log.png")


    plt = plot(B, -log.(Z), xscale=:log10, st=:scatter, dpi=500,
                    legend=false, xlabel="log(β)", ylabel="Free energy")
    savefig(plt, "FE.png")
end

function output_results(results::Vector{Result})
    num_steps = size(results[T].g)[1]
    M = size(results[T].g)[2]
    column_names = vcat(["E", "β", "K", "P"], ["X$i" for i in 1:M], )

    output = DataFrame()
    for col_name in column_names
        output[!, Symbol(col_name)] = Any[]
    end

    for i in 1:num_steps
        for t in 1:T
            gᵣ = results[t].g[i, :]
            pattern = join(string.(gᵣ))
            series = vcat(
                [results[t].E[i], results[t].β, sum(gᵣ), pattern], gᵣ
            )

            push!(output, series)
        end
    end

    CSV.write("output_result.csv",  output)
    save("output.jld2", "results", output)
end

function load_data(path_input_data, path_output_data, λ_vector::Vector{Float64})
    y_df = CSV.read(path_output_data, DataFrame)
    y::Vector{Float64} = y_df[:, 1]
    N::Int64 = length(y)

    X_df = CSV.read(path_input_data, DataFrame)
    X = Matrix{Float64}(X_df)
    X = hcat(X, ones(N))
    feature_label::Vector{String} = push!(names(X_df), "bias")

    ## 事前分布
    λ₀::Float64 = 0.01
    return Data(X, y, λ₀, λ_vector, feature_label)
end

if abspath(PROGRAM_FILE) == @__FILE__

    num_steps::Int64 = 1000 # MCMCステップ数
    T::Int64 = 16
    τ::Float64 = 1.25
    λ_vector::Vector{Float64} = Vector([10.0^i for i in 0.0:0.02:3.5])

    random_seed::Int64 = tryparse(Int64, ARGS[1])
    Random.seed!(random_seed)
    path_input_data = ARGS[2]
    path_output_data = ARGS[3]

    data::Data = load_data(path_input_data, path_output_data, λ_vector)

    B::Vector{Float64} = Vector([τ^(t-T) for t in 1:T])
    M::Int64 = length(data.labels)
    replicas = Array{Replica}(undef, T)
    for r in 1:T
        Θ = Parameters( rand([1, 0], M) )
        replicas[r] = Replica(Θ , B[r], 1e8 )
    end

    results::Vector{Result} = replica_exchange_mcmc(data, replicas, num_steps)

    output_results(results)
    
    num = fill(num_steps, T)
    num[1] = num_steps / 2.0
    num[T] = num_steps / 2.0
    plt = plot(B, 100.0*[results[r].χ for r in 1:T] ./ num, st="scatter", ylims=(0, 115), dpi=500,
                        xlabel="log(β)", ylabel="exchange ratio [%]", label="", xscale=:log10, color="gray")
    savefig(plt, "exchange_ratio.png")

    g_sampling = Vector([sum(results[T].g[:, i]) for i in 1:size(results[T].g)[2]])
    plt = plot(data.labels, 100*g_sampling / num_steps, st="bar", label="",
                        ylims=(0, 105),dpi=500,xlabel="feature labels", ylabel="probability [%]",
                        color="gray")
    savefig(plt, "g_sampling.png")

    G = Matrix{Int64}(undef, M, T)
    for t in 1:T
        gₜ = Vector([sum(results[t].g[:, i]) for i in 1:size(results[t].g)[2]])
        G[:, t] = gₜ
    end
    plt = heatmap( B, data.labels, 100.0*G./num_steps, c=:jet, size=(600, 400), dpi=500,
                        xscale=:log10, xlabel="log(β)", ylabel="feature labels")
    savefig(plt, "heatmap.png")

    
    # 列パターンごとの個数を計算
    g_df = DataFrame(results[T].g, :auto)
    pattern_counts = combine(groupby(g_df, names(g_df)), nrow)
    sort!(pattern_counts, :nrow, rev = true)
    plt = heatmap(1:size(pattern_counts)[1], 1:M, Matrix{Int64}(pattern_counts[:, 1:M])', 
                            c=:binary, grid = true, legend = false, lw = 2, dpi=500,
                            xlabel="probability [%]", ylabel="feature labels")

    plt = vline!([1.5:length(pattern_counts.nrow)+0.5], color=:gray, linewidth=2, label="", dpi=500)
    plt = hline!([1.5:M+0.5], color=:gray, linewidth=2, label="", dpi=500)

    plt = xticks!(Vector(1:length(pattern_counts.nrow)), ["$(round(i, digits=2))" for i in 100.0.*pattern_counts.nrow./num_steps])
    plt = yticks!(Vector(1:M), data.labels)

    savefig(plt, "combination.png")


    # 
    ĝ = results[T].g[argmin(results[T].E), :]
    Xₛ = data.X[:, Bool.(ĝ)]; y = data.y
    λ_vector = data.λ_vector

    Λ₀ = data.λ₀ * Matrix(I, sum(ĝ), sum(ĝ))
    ϕ = calc_marginal_likelihood_over_lambda(Xₛ, y, Λ₀, λ_vector)
    FE = integrate_over_lambda(λ_vector, ϕ)

    ϕ_max = maximum(-ϕ)
    plt = plot(λ_vector, exp.(-ϕ.-ϕ_max), st="scatter", color="gray",
            xlabel="λ", ylabel="model evidence", label="", dpi=500)
    savefig(plt, "over_lambda.png")

    λ̂ = λ_vector[argmin(ϕ)]
    μ̂, Λ̂ = calc_posterior_distribution(Xₛ, y, λ̂, Λ₀)
    ŷ = Xₛ * μ̂
    plt = plot([-4:4], [-4:4], label="", color="black")
    plt = plot!(y, ŷ, st="scatter", size=(500, 500), dpi=500, label="",
                        ylabel="predict", xlabel="true", color="gray", ms=5)
    savefig(plt, "prediction.png")
    
    calc_DoS(B, results)

end
