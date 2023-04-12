using Flux, ForwardDiff, LinearAlgebra, PrettyTables, BenchmarkTools

f(x) = 1
L = 1
u_sol(x) = (x-x^2)*0.5
u_0 = [0 for i in 1:50] |> gpu
u_1 = [0 for i in 1:50] |> gpu

measurements = [rand(Float64) for i in 1:30] |> gpu #Shouldn't be used as of right now
collocations = [rand(Float64) for i in 1:100] |> gpu;

errors = zeros(2,2)
timing = zeros(2,2)

for i=1:2
    for j=1:2
        u = Chain( x->[x],
            Dense(1 => 40,tanh),
            Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
            Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
            Dense(40 => 1),
            first) |> gpu;

        ∂2u∂x2(x) = ForwardDiff.derivative(x->ForwardDiff.derivative(u,x),x)

        loss_ode1() = sum(abs2, (-∂2u∂x2.(collocations) .- f(collocations)))/length(collocations)
        loss_data1() = sum(abs2, (u.(measurements) .- u_sol.(measurements)))/length(measurements) + sum(abs2, (u(0) .- u_0))/length(u_0) + sum(abs2, (u(L) .- u_1))/length(u_1)         #What I'm trying to avoid
        #print("\n Loss ODE1 - Loss Data1\n")
        #print(loss_ode1(), " - ", loss_data1(),"\n")

        function extra()
            a = [-∂2u∂x2(x) for x in collocations] |> gpu
            b = [f(x) for x in collocations] |> gpu
            c = [u(x) for x in measurements] |> gpu
            d = [u_sol(x) for x in measurements] |> gpu
            return ((a,b), (c,d))
        end

        loss_ode2() = Flux.mse(extra()[1][1], extra()[1][2])
        loss_data2() =  Flux.mse(u(0), u_0) + Flux.mse(u(L), u_1) + Flux.mse(extra()[2][1], extra()[2][2])
        #print("\n Loss ODE3 - Loss Data3\n")
        #print(loss_ode3(), " - ", loss_data3(),"\n")

        composed_loss1() =  loss_ode1() + 0.38*loss_data1()
        composed_loss2() =  loss_ode1() + 0.38*loss_data2()
        composed_loss3() =  loss_ode2() + 0.38*loss_data1()
        composed_loss4() =  loss_ode2() + 0.38*loss_data2()

        opt = Flux.Adam()
        data = Iterators.repeated((), 5000)
        iter = 0

            if i==1 && j==1
                #print("\n composed_loss1\n")
                #print(composed_loss1(),"\n")
                timing[i,j] = median(@benchmark Flux.train!($composed_loss1, $Flux.params($u), $data, $opt)).time
                #Flux.train!(composed_loss1, Flux.params(u), data, opt)
                #print(composed_loss1(),"\n")
            elseif i==1 && j==2
                #print("\n composed_loss2\n")
                #print(composed_loss2(),"\n")
                timing[i,j] = median(@benchmark Flux.train!($composed_loss2, $Flux.params($u), $data, $opt)).time
                #Flux.train!(composed_loss2, Flux.params(u), data, opt)
                #print(composed_loss2(),"\n")
            elseif i==2 && j==1
                #print("\n composed_loss3\n")
                #print(composed_loss3(),"\n")
                timing[i,j] = median(@benchmark Flux.train!($composed_loss3, $Flux.params($u), $data, $opt)).time
                #Flux.train!(composed_loss3, Flux.params(u), data, opt)
                #print(composed_loss3(),"\n")
            else
                #print("\n composed_loss4\n")
                #print(composed_loss4(),"\n")
                timing[i,j] = median(@benchmark Flux.train!($composed_loss4, $Flux.params($u), $data, $opt)).time
                #Flux.train!(composed_loss4, Flux.params(u), data, opt)
                #print(composed_loss4(),"\n")
            end

        plot_t = 0:0.01:L
        learned_plot = u.(plot_t) |> gpu
        real_plot = u_sol.(plot_t) |> gpu
        error = norm(learned_plot - real_plot)
        errors[i,j] = error
    end
end

header = ([" ", "My Data", "Flux Data"])
header_c = ["My ODE", "Flux ODE"]
pretty_table([header_c errors[:,1] errors[:,2]], backend = Val(:latex), header=header)
pretty_table([header_c timing[:,1] timing[:,2]], backend = Val(:latex), header=header)