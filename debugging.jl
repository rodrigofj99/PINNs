using Flux, LinearAlgebra, BenchmarkTools, Plots, ForwardDiff, ReverseDiff, FiniteDifferences

f(x) = 1
L = 1
u_sol(x) = (x-x^2)*0.5
u_0 = [0 for i in 1:1] 
u_1 = [0 for i in 1:1] 

measurements = [rand(Float64) for i in 1:30]  #Shouldn't be used as of right now
collocations = [rand(Float64) for i in 1:100] ;

u = Chain( x->[x],
    Dense(1 => 40,tanh),
    Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
    Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),Dense(40 => 40,tanh),
    Dense(40 => 1),
    first) ;

    ϵ = sqrt(eps(Float64))

function loss_func(NN)
   print(u(0.25),"\n")
   #d2udx2(NN,x) = ForwardDiff.derivative(x->ForwardDiff.derivative(NN,x),x)
   #p, re = Flux.destructure(NN)
   #a = [-d2udx2(NN,x) for x in collocations]
   #a = ReverseDiff.gradient(NN,collocations) 
   #a = [-Flux.jacobian(NN,x)[1] for x in global collocations] 
   a = [-(NN(x+ϵ)-2*NN(x)+NN(x-ϵ))/ϵ^2 for x in collocations] 
   #a = [-(NN(x+ϵ)-NN(x))/ϵ for x in collocations] 
   #_epsilon = inv(first(ϵ[ϵ .!= zero(ϵ)]))
   #a = [-(NN(x .+ ϵ) .+ NN(x .- ϵ) .- 2 .* NN(x)) .* _epsilon^2 for x in collocations] 
   #print(a,"\n")
   b = [f(x) for x in collocations] 
   #c = [NN(x) for x in measurements] 
   #d = [u_sol(x) for x in measurements] 
   loss = Flux.mse(a,b)
   #loss += 0.1*(Flux.mse(NN(0), u_0) + Flux.mse(NN(L), u_1))# + Flux.mse(c,d))
   return loss
end
#=function loss_func()
   d2udx2(x) = ForwardDiff.derivative(x->ForwardDiff.derivative(u,x),x)
   a = [-d2udx2(x) for x in collocations] 
   b = [f(x) for x in collocations] 
   c = [u(x) for x in measurements] 
   d = [u_sol(x) for x in measurements] 
   loss = Flux.mse(a,b)
   #loss += 0.38*(Flux.mse(u(0), u_0) + Flux.mse(u(L), u_1) + Flux.mse(c,d))
   return loss
end=#

data = Iterators.repeated((), 50)
opt_state = Flux.setup(Flux.Adam(), u)
Flux.train!(loss_func, u, data, opt_state)

#opt = Flux.Adam()
#Flux.train!(loss_func, Flux.params(u), data, opt)

plot_t = 0:0.01:L

learned_plot = u.(plot_t) 
real_plot = u_sol.(plot_t) 

error = norm(learned_plot - real_plot)

display("The error is $error")

plot(plot_t,real_plot,xlabel="x",label="True")
plot!(plot_t,learned_plot,label="NN")

#savefig(q,"try2")