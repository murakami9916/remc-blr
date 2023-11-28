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
using GR

GR.inline("png")

### 構造体
# -----------------------------------------
struct Data
    X::Matrix{Float64}
    y::Vector{Float64}
    λ₀::Float64
    λ_vector::Vector{Float64}
    neg_log_prior_density::Vector{Float64}
    labels::Vector{String}
    K::Int64
    Data(X, y, λ₀, λ_vector, neg_log_prior_density, labels, K) = new(X, y, λ₀,λ_vector, neg_log_prior_density, labels, K)
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

@inline function sampling_indicator(M::Int64, K::Int64)
    g::Vector{Int64} = zeros(M)
    g[rand(1:M, K)] .= 1
    return g
end

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
    Threads.@threads for m in 1:M
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

    ## 事前確率の足し合わせ
    ϕ = ϕ + data.neg_log_prior_density
    
    FE::Float64 = integrate_over_lambda(λ_vector, ϕ)

    return FE
end

function metropolis_indicator(data::Data, replica::Replica)
    L::Int64 = length(replica.Θ.g)
    β::Float64 = replica.β
    K = data.K

    for i in randperm(L)
        Θₜ = deepcopy(replica.Θ)
        
        # 0/1の反転
        Θₜ.g[i] = ( ~Θₜ.g[i] + 2 )

        if sum(Θₜ.g) > K
            continue
        end

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

function calc_DoS(num_steps::Int64, K::Int64, B::Vector{Float64}, results::Vector{Result}, output_path::String)

    # マルチヒストグラムによるDoSの計算
    T = length(B)
    M = size(results[T].g)[2]
    num_states = sum([binomial(M,k_i) for k_i in 0:K])
    E_all = Array{Float64}(undef, num_steps, T)
    bottom = 1e-1
    for r in 1:T
        E_all[:, r] = results[r].E
    end
    all_neg_log_likelihood = vec(E_all)
    n = size(E_all)[1] 
    hist = fit(Histogram, all_neg_log_likelihood, nbins=100)

    edges = Vector(hist.edges[1])
    E = (edges[1:end-1] + edges[2:end]) ./ 2.0
    H = hist.weights
    tolerance = 1e-8
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

    plt = Plots.bar( E[D.!=0], num_states .* ( D[D.!=0] / sum(D[D.!=0]) ), legend=false, 
            xlabel="Negative marginal log-likelihood", ylabel = "Density of states",
            dpi=500, color="gray", fillrange=bottom)
    Plots.savefig(plt, "$output_path/DoS_linear.png")


    plt = Plots.bar( E[D.!=0], num_states .* ( D[D.!=0] / sum(D[D.!=0]) ), legend=false, 
            xlabel="Negative marginal log-likelihood", ylabel = "Density of states",yscale=:log10,
            dpi=500, color="gray", fillrange=bottom)
    # savefig(plt, "$output_path/DoS_log.png")
    png(plt, "$output_path/DoS_log.png")


    plt = Plots.plot(B, -log.(Z), xscale=:log10, st=:scatter, dpi=500,
                    legend=false, xlabel="log(β)", ylabel="Free energy")
    # savefig(plt, "$output_path/FE.png")
    png(plt, "$output_path/FE.png")

    CSV.write( "$output_path/free_energy.csv",  DataFrame(B=B, FE=-log.(Z), Z=Z) )
end

function output_results(data::Data, T::Int64, results::Vector{Result}, output_path::String)
    num_steps = size(results[T].g)[1]
    M = size(results[T].g)[2]
    column_names = vcat(["E", "β", "K", "P"], data.labels)

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

    CSV.write("$output_path/output_result.csv",  output)
    save("$output_path/output.jld2", "results", output)
end

function plot_result(results::Vector{Result}, data::Data, num_steps::Int64,
                                    B::Vector{Float64}, M::Int64, T::Int64, output_path::String)

    num = fill(num_steps, T)
    num[1] = num_steps / 2.0
    num[T] = num_steps / 2.0
    plt = Plots.plot(B, 100.0*[results[r].χ for r in 1:T] ./ num, st="scatter", ylims=(0, 115), dpi=500,
                        xlabel="log(β)", ylabel="exchange ratio [%]", label="", xscale=:log10, color="gray")
    Plots.savefig(plt, "$output_path/exchange_ratio.png")

    g_sampling = Vector([sum(results[T].g[:, i]) for i in 1:size(results[T].g)[2]])
    plt = Plots.plot(1:length(data.labels), 100*g_sampling / num_steps, st="bar", label="",
                        ylims=(0, 105),dpi=500,xlabel="feature labels", ylabel="probability [%]",
                        color="gray", xrotation=90, xticks=(1:length(data.labels), data.labels))

    Plots.savefig(plt, "$output_path/g_sampling.png")

    G = Matrix{Int64}(undef, M, T)
    for t in 1:T
        gₜ = Vector([sum(results[t].g[:, i]) for i in 1:size(results[t].g)[2]])
        G[:, t] = gₜ
    end
    plt = Plots.heatmap( B, data.labels, 100.0*G./num_steps, c=:jet, size=(600, 400), dpi=500,
                        xscale=:log10, xlabel="log(β)", ylabel="feature labels", clim=(0, 100),
                        yticks=(0.5:length(data.labels), data.labels))
    Plots.savefig(plt, "$output_path/heatmap.png")

    
    # 列パターンごとの個数を計算
    g_df = DataFrame(results[T].g, :auto)
    pattern_counts = combine(groupby(g_df, names(g_df)), nrow)
    sort!(pattern_counts, :nrow, rev = true)
    # pattern_counts = pattern_counts[:, 1:minimum(5, length(pattern_counts.nrow))]
    num_c = minimum([15, length(pattern_counts.nrow)])
    pattern_counts = pattern_counts[1:num_c, :]
    plt = Plots.heatmap(1:num_c, 1:M, Matrix{Int64}(pattern_counts[:, 1:M])', 
                            c=:binary, grid = true, legend = false, lw = 2, dpi=500,
                            xlabel="probability [%]", ylabel="feature labels")

    plt = vline!([1.5:num_c+0.5], color=:gray, linewidth=2, label="", dpi=500)
    plt = hline!([1.5:M+0.5], color=:gray, linewidth=2, label="", dpi=500)

    plt = xticks!(Vector(1:num_c),
                         ["$(round(i, digits=2))" for i in 100.0.*pattern_counts.nrow./num_steps])
    plt = yticks!(Vector(1:M), data.labels)

    Plots.savefig(plt, "$output_path/combination.png")


    # 
    ĝ = results[T].g[argmin(results[T].E), :]
    Xₛ = data.X[:, Bool.(ĝ)]; y = data.y
    λ_vector = data.λ_vector

    Λ₀ = data.λ₀ * Matrix(I, sum(ĝ), sum(ĝ))
    ϕ = calc_marginal_likelihood_over_lambda(Xₛ, y, Λ₀, λ_vector)
    FE = integrate_over_lambda(λ_vector, ϕ)

    ϕ_max = maximum(-ϕ)
    area = trapezoidal_integration( λ_vector, exp.(-ϕ.-ϕ_max) )
    λ_posterior_distribution = exp.(-ϕ.-ϕ_max) / area
    plt = Plots.plot(λ_vector, λ_posterior_distribution, xscale=:log10, label="Posterior distribution",
                        lw = 2, fill=(0, :gray), color=:black, alpha=0.8, xlabel="precision parameter λ", ylabel="Probability", dpi=500)
    plt = Plots.plot!(λ_vector, exp.(-data.neg_log_prior_density), xscale=:log10, label="Prior distribution",
                        lw = 2, fill=(0, :gray), color=:black, alpha=0.2)
    Plots.savefig(plt, "$output_path/over_lambda.png")

    λ̂ = λ_vector[argmin(ϕ)]
    μ̂, Λ̂ = calc_posterior_distribution(Xₛ, y, λ̂, Λ₀)
    ŷ = Xₛ * μ̂
    plt = Plots.plot([-4:4], [-4:4], label="", color="black")
    plt = Plots.plot!(y, ŷ, st="scatter", size=(500, 500), dpi=500, label="",
                        ylabel="predict", xlabel="true", color="gray", ms=5)
    Plots.savefig(plt, "$output_path/prediction.png")

    f = open( "$output_path/info.txt", "w")
    println(f, "free energy"); println(f, round(FE, digits=4))
    println(f, "λ̂"); println(f, round(λ̂, digits=4))
    println(f, "ĝ"); println(f, data.labels); println(f, ĝ)
    println(f, "μ̂"); println(f, data.labels[Bool.(ĝ)]); println(f, round.(μ̂, digits=4))
    println(f, "Λ̂"); Base.print_array(f, round.(Λ̂, digits=4))
    close(f)

end

function load_data(path_input_data, path_output_data,
                                    λ_vector::Vector{Float64}, K::Int64)
    y_df = CSV.read(path_output_data, DataFrame)
    y::Vector{Float64} = y_df[:, 1]
    N::Int64 = length(y)

    X_df = CSV.read(path_input_data, DataFrame)
    X = Matrix{Float64}(X_df)
    X = hcat(X, ones(N))
    feature_label::Vector{String} = push!(names(X_df), "bias")

    # ノーマライズ
    y = ( y .- mean(y) ) ./ std(y)
    X = ( X .- mean(X, dims=1) ) ./ max.(std(X, dims=1), 1e-8)

    ## 事前分布
    λ₀::Float64 = 0.1
    λ_prior_distribution = Gamma(1.1, 50)
    neg_log_prior_density = -logpdf.(λ_prior_distribution, λ_vector)
    return Data(X, y, λ₀, λ_vector, neg_log_prior_density, feature_label, K)
end

if abspath(PROGRAM_FILE) == @__FILE__

    # setting
    num_steps::Int64 = 100 # MCMCステップ数
    T::Int64 = 32
    τ::Float64 = 1.1 # 1.2
    K::Int64 = 5
    λ_vector::Vector{Float64} = Vector([10.0^i for i in -1:0.025:2.5])

    random_seed::Int64 = tryparse(Int64, ARGS[1])
    Random.seed!(random_seed)
    path_input_data = ARGS[2]
    path_output_data = ARGS[3]

    input_basename = split(basename(path_input_data), ".")[1]
    output_basename = split(basename(path_output_data), ".")[1]
    output_path = "$output_basename/$input_basename/$random_seed"
    
    data::Data = load_data(path_input_data, path_output_data, λ_vector, K)

    B::Vector{Float64} = Vector([τ^(t-T) for t in 1:T])
    M::Int64 = length(data.labels)
    replicas = Array{Replica}(undef, T)
    g = sampling_indicator(M, K)
    for r in 1:T
        Θ = Parameters( g )
        replicas[r] = Replica(Θ , B[r], 1e8 )
    end
    results::Vector{Result} = replica_exchange_mcmc(data, replicas, num_steps)

    mkpath(output_path)
    output_results(data, T, results, output_path)
    calc_DoS(num_steps, K, B, results, output_path)
    plot_result(results, data, num_steps, B, M, T, output_path)
    
end
