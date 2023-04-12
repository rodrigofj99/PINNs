using Flux, QuadGK, Plots, ForwardDiff, LinearAlgebra, CUDA

####################################################################
#-∂[κ(x)∂[u(x)]] = f(x)
####################################################################

struct Split{T}
  paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

f(x) = x
κ(x) = x
u_sol(x) = (x-x^2)*0.5
L = 1
u_0 = zeros(1,50)
u_1 = zeros(1,50)
measurements = [rand() for x in 1:30]


NN = Chain( x->[x],
           Dense(1 => 40,tanh),
           Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
           Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
           Split(Dense(40 => 1), Dense(40 => 1)))
           
u(x) = NN(x)[1][1]
∂u∂x(x) = ForwardDiff.derivative(u,x)
k(x) = NN(x)[2][1]

temp(x) = ∂u∂x(x)*k(x)
collocations = [rand() for i in 1:100]

λ = 0.01

H(x) = ForwardDiff.derivative(temp,x)

a = [-H(x) for x in collocations] |> gpu
b = [f(x) for x in collocations] |> gpu

loss_ode() = sum((a.-b).^2)/length(collocations)
loss_data() = sum(abs2, (u(x) - u_sol(x) for x in measurements))/length(measurements) + sum(abs2, (u(0)-u_0[i] for i in eachindex(u_0)))/length(u_0) + sum(abs2, (u(L)-u_1[i] for i in eachindex(u_1)))/length(u_1)         #What I'm trying to avoid

composed_loss() = loss_data() + λ*loss_ode()


#Flux.trainable = (Flux.params(NN),k)
#Flux.params!

opt = Flux.Adam()
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(composed_loss())
  end
end
display(composed_loss())
Flux.train!(composed_loss, Flux.params(NN,k), data, opt; cb=cb)


plot_t = 0:0.01:L

learned_plot_u = u.(plot_t)
learned_plot_k = k.(plot_t)
real_plot = u_sol.(plot_t)

error_u = norm(learned_plot_u - real_plot)
error_k = norm(learned_plot_k - real_plot)

display("The error in u(x) is $error_u")
display("The error in κ(x) is $error_k")

plot(plot_t,real_plot,xlabel="x",label="True")
display(plot!(plot_t,learned_plot_u,label="NN"))

plot(plot_t,κ.(plot_t),xlabel="x",label="True")
plot!(plot_t,learned_plot_k,label="NN")
