using Flux, ForwardDiff, LinearAlgebra, CUDA, BenchmarkTools

####################################################################
#-∂[κ(x)∂[u(x)]] = f(x)
#κ(x) = 1
####################################################################

function heatEq(activation, optimizer, bestParams, errors, index1, index2, timing, λ, learning_rate=1, c=1)
#function heatEq(activation, optimizer, bestParams, errors, index1, index2, timing, λ, learning_rate=1, c=1)
    f(x) = 1
    L = 1
    u_sol(x) = (x-x^2)*0.5
    u_0 = [0 for i in 1:50] |> gpu
    u_1 = [0 for i in 1:50] |> gpu

    measurements = [rand(Float64) for i in 1:30] |> gpu #Shouldn't be used as of right now

    #### Not using right now ####
    #if activation == "tanh"    
        u = Chain( x->[x],
                Dense(1 => 40,tanh),
                Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
                Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
                Dense(40 => 1),
                first) |> gpu
    #end

    collocations = [rand(Float64) for i in 1:100] |> gpu


    ∂2u∂x2(x) = ForwardDiff.derivative(x->ForwardDiff.derivative(u,x),x)
    a = [-∂2u∂x2(x) for x in collocations] |> gpu
    b = [f(x) for x in collocations] |> gpu
    loss_data() = sum(abs2, (u(x) - u_sol(x) for x in measurements))/length(measurements) + sum(abs2, (u(0)-u_0[i] for i in eachindex(u_0)))/length(u_0) + sum(abs2, (u(L)-u_1[i] for i in eachindex(u_1)))/length(u_1)         #What I'm trying to avoid
    loss_ode() = sum((a.-b).^2)/length(collocations)
    composed_loss() =  loss_ode() + λ*loss_data()             #I tried a multiplying either term with λ but the outcome is the same. 
                                                            #When I only use loss_ode and NOT loss_data I a get a somewhat better result, but still not good
    
    #### Used when tunning ####
    function smart_loss()
        loss = composed_loss()
        if loss == -Inf  # everything perfect, stop now
            #print("Loss in -Inf\n")
            Flux.stop()   # This throws a StopException, which Flux.train! catches and treats an an termination signal
        elseif loss == Inf  # This is bad, need to fix the model, optimizer can't help
            #throw(DomainError(loss))  # this will bubble up and outside train! to somewhere it can be delt with
            #print("Loss in Inf\n")
            Flux.stop()
        elseif isnan(loss)  # it is just funky some times, ignore it.
            #print("Loss in -Inf\n")
            return 0  # this s an untracked value, so no gradients will propergate through it. So model will not change.
        else
            return loss  # A tracked value. This can be used by optimizer to update state.
        end
    end 


    if optimizer == "Momentum"
        opt = Flux.Momentum(learning_rate,c)
    elseif optimizer == "Nesterov"
        opt = Flux.Nesterov(learning_rate,c)
    elseif optimizer == "AdamW"
        opt = Flux.AdamW(learning_rate,(0.9, 0.999),c)
    elseif optimizer == "RMSProp"
        opt = Flux.RMSProp(learning_rate)
    elseif optimizer == "Adam"
        opt = Flux.Adam(learning_rate)
    elseif optimizer == "RAdam"
        opt = Flux.RAdam(learning_rate)
    elseif optimizer == "AdaMax"
        opt = Flux.AdaMax(learning_rate)
    elseif optimizer == "OAdam"
        opt = Flux.OAdam(learning_rate)
    elseif optimizer == "AdaBelief"
        opt = Flux.AdaBelief(learning_rate)
    elseif optimizer == "AdaGrad"
        opt = Flux.AdaGrad()
    elseif optimizer == "AdaDelta"
        opt = Flux.AdaDelta()
    elseif optimizer == "AMSGrad"
        opt = Flux.AMSGrad()
    else
        opt = Flux.NAdam()
    end

    data = Iterators.repeated((), 5000)
    iter = 0
    cb = function ()
        global iter += 1
        if iter % 500 == 0
            display(composed_loss())
        end
    end

    tmp = median(@benchmark Flux.train!($smart_loss, $Flux.params($u), $data, $opt)).time
    #Flux.train!(smart_loss, Flux.params(u), data, opt)

    grid = 0:0.01:L
    learned_plot = u.(grid) |> gpu
    real_plot = u_sol.(grid) |> gpu
    error = norm(learned_plot - real_plot)
    
    if (error < errors[index1])
        bestParams[index1] = [λ, learning_rate, c]
        errors[index1] = error
        timing[index1] = tmp
    end

end