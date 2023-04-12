using Flux, ForwardDiff, LinearAlgebra, BenchmarkTools, PrettyTables

f(x) = 1
L = 1
u_sol(x) = (x-x^2)*0.5
u_0 = [0 for i in 1:1] |> gpu
u_1 = [0 for i in 1:1] |> gpu

measurements = [rand(Float64) for i in 1:30] |> gpu #Shouldn't be used as of right now
collocations = [rand(Float64) for i in 1:100] |> gpu;

errors = [Inf64, Inf64]
timing = [Inf64, Inf64]


###################### OLD VERSION ##########################

u = Chain( x->[x],
    Dense(1 => 40,tanh),
    Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
    Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
    Dense(40 => 1),
    first) |> gpu;

∂2u∂x2(x) = ForwardDiff.derivative(x->ForwardDiff.derivative(u,x),x)
function extra()
    a = [-∂2u∂x2(x) for x in collocations] |> gpu
    b = [f(x) for x in collocations] |> gpu
    c = [u(x) for x in measurements] |> gpu
    d = [u_sol(x) for x in measurements] |> gpu
    return ((a,b), (c,d))
end
loss_ode() = Flux.mse(extra()[1][1], extra()[1][2])
loss_data() =  Flux.mse(u(0), u_0) + Flux.mse(u(L), u_1) + Flux.mse(extra()[2][1], extra()[2][2])
composed_loss() =  loss_ode() + 0.38*loss_data()

opt = Flux.Adam()
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(composed_loss())
    end
end

timing[1] =  median(@benchmark Flux.train!(composed_loss, Flux.params(u), data, opt)).time

plot_t = 0:0.01:L

learned_plot = u.(plot_t) |> gpu
real_plot = u_sol.(plot_t) |> gpu

errors[1] = norm(learned_plot - real_plot)

#=display("The error is $error")

plot(plot_t,real_plot,xlabel="x",label="True")
plot!(plot_t,learned_plot,label="NN")
#scatter!(measurements,u_sol.(measurements),label="Measurements")=#



####################### NEW VERSION ##########################

u2 = Chain( x->[x],
    Dense(1 => 40,tanh),
    Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
    Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
    Dense(40 => 1),
    first) |> gpu;

d2udx2(NN,x) = ForwardDiff.derivative(x->ForwardDiff.derivative(NN,x),x)
function extra2(NN)
    a = [-d2udx2(NN,x) for x in collocations] |> gpu
    b = [f(x) for x in collocations] |> gpu
    c = [u(x) for x in measurements] |> gpu
    d = [u_sol(x) for x in measurements] |> gpu
    return ((a,b), (c,d))
end
loss_ode2(NN) = Flux.mse(extra2(NN)[1][1], extra2(NN)[1][2])
loss_data2(NN) =  Flux.mse(NN(0), u_0) + Flux.mse(NN(L), u_1) + Flux.mse(extra2(NN)[2][1], extra2(NN)[2][2])
composed_loss2(NN) =  loss_ode2(NN) + 0.38*loss_data2(NN)

opt_state = Flux.setup(Flux.Adam(), u2)
timing[2] = median(@benchmark Flux.train!(composed_loss2, u2, data, opt_state)).time

learned_plot = u.(plot_t) |> gpu
real_plot = u_sol.(plot_t) |> gpu

errors[2] = norm(learned_plot - real_plot)

header = ([" ", "Old Version", "New Version"])
header_c = ["Error", "Timing"]
pretty_table([header_c [errors[1], timing[1]] [errors[2], timing[2]]], backend = Val(:latex), header=header)