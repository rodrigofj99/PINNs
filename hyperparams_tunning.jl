using Hyperopt, Optim, PrettyTables

function RandomSampler_3(activation, optimizer, bestParams,errors,i,j, timing)
    ho = @hyperopt for k=1, sampler = RandomSampler(),
        a = LinRange(0.001,1,100),
        b = LinRange(0.001,1,100),
        c = LinRange(0.001,1,100)
        heatEq(activation, optimizer, bestParams, errors, i, j, timing, a, b, c)
        end
    #=bestParams[i][j] = (ho.minimizer[1], ho.minimizer[2], ho.minimizer[3])
    print(bestParams[i][j])
    if (ho.minimum == Nothing)
        errors[i] = Inf64
    else
        errors[i] = ho.minimum
    end=#
end

function LHSampler_3(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for k=1, sampler = LHSampler(),
        a = LinRange(0.001,1,100),
        b = LinRange(0.001,1,100),
        c = LinRange(0.001,1,100)
        heatEq(activation, optimizer, a, b, c)
        end
    bestParams[i][j] = (ho.minimizer[1], ho.minimizer[2], ho.minimizer[3])
    if (ho.minimum == Nothing)
        errors[i] = Inf64
    else
        errors[i] = ho.minimum
    end
end

function Hyperband_3(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for resources=1, sampler=Hyperband(R=1, η=3, inner=RandomSampler()),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(0.001,1,1000),
        b = LinRange(0.001,1,100),
        c = exp10.(LinRange(0.001,1,10000))
        if state !== nothing
            algorithm, x0 = state
        else
            x0 = [a,b,c]
        end
        res = Optim.optimize(x->heatEq(activation, optimizer, x[1], x[2], x[3]), x0, algorithm, Optim.Options(time_limit=resources+1, show_trace=false))
        #bestParams[i][j] = (algorithm, (Optim.minimizer(res)[1], Optim.minimizr(res)errors[i] =[2], Optim.minimizer(res)[3])), Optim.minimum(res) 
        bestParams[i][j] = (Optim.minimizer(res)[1], Optim.minimizer(res)[2], Optim.minimizer(res)[3]),
        if (Optim.minimum(res) == Nothing)
            errors[i] = Inf64
        else
            errors[i] = Optim.minimum(res) 
        end
    end
end

function BOHB_3(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for resources=1, sampler=Hyperband(R=1, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(0.001,1,1000),
        b = LinRange(0.001,1,100),
        c = exp10.(LinRange(0.001,1,10000))
        if state !== nothing
            algorithm, x0 = state
        else
            x0 = [a,b,c]
        end
        res = Optim.optimize(x->heatEq(activation, optimizer, x[1], x[2], x[3]), x0, algorithm, Optim.Options(time_limit=resources+1, show_trace=false))
        bestParams[i][j] = (Optim.minimizer(res)[1], Optim.minimizer(res)[2], Optim.minimizer(res)[3]),
        if (Optim.minimum(res) == Nothing)
            errors[i] = Inf64
        else
            errors[i] = Optim.minimum(res) 
        end
    end
end


function RandomSampler_2(activation, optimizer, bestParams,errors,i,j,timing)
    ho = @hyperopt for k=1, sampler = RandomSampler(),
        a = LinRange(0.001,1,100),
        b = LinRange(0.001,1,100)
        heatEq(activation, optimizer, bestParams, errors, i, j, timing, a, b)
        end
    #=bestParams[i][j] = (ho.minimizer[1], ho.minimizer[2], 0.0)
    if (ho.minimum == Nothing)
        errors[i] = Inf64
    else   
        errors[i] = ho.minimizer
    end=#
end

function LHSampler_2(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for k=1, sampler = LHSampler(),
        a = LinRange(0.001,1,100),
        b = LinRange(0.001,1,100)
        heatEq(activation, optimizer, a, b)
        end
    bestParams[i][j] = (ho.minimizer[1], ho.minimizer[2], 0.0)
    if (ho.minimum == Nothing)
        errors[i] = Inf64
    else
        errors[i] = ho.minimizer
    end
end

function Hyperband_2(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for resources=1, sampler=Hyperband(R=1, η=3, inner=RandomSampler()),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(0.001,1,1000),
        b = LinRange(0.001,1,100)
        if state !== nothing
            algorithm, x0 = state
        else
            x0 = [a,b]
        end
        res = Optim.optimize(x->heatEq(activation, optimizer, x[1], x[2]), x0, algorithm, Optim.Options(time_limit=resources+1, show_trace=false))
        bestParams[i][j] = (Optim.minimizer(res)[1], Optim.minimizer(res)[2], 0.0)
        if (Optim.minimum(res) == Nothing)
            errors[i] = Inf64
        else
            errors[i] = Optim.minimum(res) 
        end
    end
end

function BOHB_2(activation, optimizer, bestParams,errors,i,j)
        ho = @hyperopt for resources=1, sampler=Hyperband(R=1, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(0.001,1,1000),
        b = LinRange(0.001,1,100)
        if state !== nothing
            algorithm, x0 = state
        else
            x0 = [a,b]
        end
        res = Optim.optimize(x->heatEq(activation, optimizer, x[1], x[2]), x0, algorithm, Optim.Options(time_limit=resources+1, show_trace=false))
        bestParams[i][j] = (Optim.minimizer(res)[1], Optim.minimizer(res)[2], 0.0)
        if (Optim.minimum(res) == Nothing)
            errors[i] = Inf64
        else
            errors[i] = Optim.minimum(res) 
        end
    end
end



function RandomSampler_1(activation, optimizer, bestParams,errors,i,j,timing)
    ho =  @hyperopt for k=1, sampler = RandomSampler()
        a = LinRange(0.001,1,100)
        heatEq(activation, optimizer, bestParams, errors, i, j, timing, a)
        
    end
    #=bestParams[i][j] = (ho.minimizer[1], 0.0, 0.0)
    if (ho.minimum == Nothing)
        errors[i] = Inf64
    else
        errors[i] = ho.minimizer
    end=#
end

function LHSampler_1(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for k=1, sampler = LHSampler(),
        a = LinRange(0.001,1,100)
        heatEq(activation, optimizer, a)
        end
    bestParams[i][j] = (ho.minimizer[1], 0.0, 0.0)
    if (ho.minimum == Nothing)
        errors[i] = Inf64
    else
        errors[i] = ho.minimizer
    end
end

function Hyperband_1(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for resources=1, sampler=Hyperband(R=1, η=3, inner=RandomSampler()),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(0.001,1,1000)
        if state !== nothing
            algorithm, x0 = state
        else
            x0 = [a]
        end
        res = Optim.optimize(x->heatEq(activation, optimizer, x[1]), x0, algorithm, Optim.Options(time_limit=resources+1, show_trace=false))
        bestParams[i][j] = (Optim.minimizer(res)[1], 0.0 ,0.0)
        if (Optim.minimum(res) == Nothing)
            errors[i] = Inf64
        else
        errors[i] = Optim.minimum(res) 
        end
    end
end

function BOHB_1(activation, optimizer, bestParams,errors,i,j)
    ho = @hyperopt for resources=1, sampler=Hyperband(R=1, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(0.001,1,1000)
        if state !== nothing
            algorithm, x0 = state
        else
            x0 = [a]
        end
        res = Optim.optimize(x->heatEq(activation, optimizer, x[1]), x0, algorithm, Optim.Options(time_limit=resources+1, show_trace=false))
        bestParams[i][j] = (Optim.minimizer(res)[1], 0.0, 0.0)
        if (Optim.minimum(res) == Nothing)
            errors[i] = Inf64
        else
            errors[i] = Optim.minimum(res) 
        end
    end
end


#CODE NOT OPTIMAL AT ALL, BUT TRYING TO MAKE IT WORK FIRST


#function hyperparams()
    include("heatEq_for_tunning.jl")
    #bestParams = [[(0.0, 0.0, 0.0) for j in 1:4] for i in 1:13]
    bestParams = [[0.0, 0.0, 0.0] for i in 1:13]
    errors = [Inf64 for i=1:13]
    timing = [Inf64 for i=1:13]
    #=for i in 1:n          #Not using right now
        if i == 1    
            activation = "..."
        elseif i == 2
            activation = ""
        elseif i == 3
            activation = ""
        else
            ...=#
        activation = "tanh"
        count = 1
        for j in 1:2####3 #Three types of optimizers (1,2,and 3 hyperparams)
            if j == 1    
                for k in 1:3 #optimizers with 3 hyperparams
                    if k == 1
                        optimizer = "Momentum"
                        RandomSampler_3(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_3(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_3(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_3(activation,optimizer,bestParams,errors,count,4)
                    elseif k == 2
                        optimizer = "Nesterov"
                        RandomSampler_3(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_3(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_3(activation,optimizer,bestParams,errors,count,3)
                        #OHB_3(activation,optimizer,bestParams,errors,count,4)
                    else
                        optimizer = "AdamW"
                        RandomSampler_3(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_3(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_3(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_3(activation,optimizer,bestParams,errors,count,4)
                    end
                end

            elseif j == 2
                for k in 1:6 #optimizers with 2 hyperparams
                    if k == 1
                        optimizer = "RMSProp"
                        RandomSampler_2(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_2(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_2(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_2(activation,optimizer,bestParams,errors,count,4)
                    elseif k == 2
                        optimizer = "Adam"
                        RandomSampler_2(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_2(activation,optimizer,bestParams,errors,count,2)
                        #yperband_2(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_2(activation,optimizer,bestParams,errors,count,4)
                    elseif k == 3
                        optimizer = "RAdam"
                        RandomSampler_2(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_2(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_2(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_2(activation,optimizer,bestParams,errors,count,4)
                    elseif k == 4
                        optimizer = "AdaMax"
                        RandomSampler_2(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_2(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_2(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_2(activation,optimizer,bestParams,errors,count,4)
                    elseif k == 5
                        optimizer = "OAdam"
                        RandomSampler_2(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_2(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_2(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_2(activation,optimizer,bestParams,errors,count,4)
                    else 
                        optimizer = "AdaBelief"
                        RandomSampler_2(activation,optimizer,bestParams,errors,count,1,timing)
                        global count +=1
                        #LHSampler_2(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_2(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_2(activation,optimizer,bestParams,errors,count,4)
                    end
                end               
                
            else
                for k in 1:4 #optimizers with 1 hyperparam (namely the one in the loss function with the PDE)
                    if k == 1
                        optimizer = "AdaGrad"
                        RandomSampler_1(activation,optimizer,bestParams,errors,count,1,timing)
                        #LHSampler_1(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_1(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_1(activation,optimizer,bestParams,errors,count,4)
                        
                    elseif k == 2
                        optimizer = "AdaDelta"
                        RandomSampler_1(activation,optimizer,bestParams,errors,count,1,timing)
                        #LHSampler_1(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_1(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_1(activation,optimizer,bestParams,errors,count,4)
                        
                    elseif k == 3
                        optimizer = "AMSGrad"
                        RandomSampler_1(activation,optimizer,bestParams,errors,count,1,timing)
                        #LHSampler_1(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_1(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_1(activation,optimizer,bestParams,errors,count,4)
                        
                    else
                        optimizer = "NAdam"
                        RandomSampler_1(activation,optimizer,bestParams,errors,count,1,timing)
                        #LHSampler_1(activation,optimizer,bestParams,errors,count,2)
                        #Hyperband_1(activation,optimizer,bestParams,errors,count,3)
                        #BOHB_1(activation,optimizer,bestParams,errors,count,4)
                        
                    end
                end
            end
        end
        #Output code for table in LaTex
        #bestParams = mapreduce(permutedims, vcat, bestParams)     #Useful to convert from array of arrays to matrix (but not needed in this case)
        hyper = ["Alpha", "Learning Rate", "c"]
        optimizers = ["Momentum", "Nesterov", "AdamW", "RMSProp", "Adam", "RAdam","AdaMax", "OAdam", "AdaBelief", "AdaGrad", "AdaDelta", "AMSGrad", "NAdam"]
        header1 = (["Optimizer", "Timing"])
        header2 = (["Optimizer", "Error"])
        pretty_table([optimizers timing], backend = Val(:latex), header=header1)
        pretty_table([optimizers errors], backend = Val(:latex), header=header2)
        pretty_table([hyper bestParams[1] bestParams[2] bestParams[3] bestParams[4] bestParams[5] bestParams[6] bestParams[7] bestParams[8] bestParams[9] bestParams[10] bestParams[11] bestParams[12] bestParams[13]], backend = Val(:latex), header=(pushfirst!(optimizers, " ")))
    #end
    #return bestParams, errors, timing
#end
