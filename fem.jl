using QuadGK, Plots, Trapz


Stiffness_Matrix = function (N)
    h = 1/(N+1)
    K = zeros(N,N)
    for j in 1:N
        K[j,j] = 2/h
    end
    for j in 1:N-1
        K[j,j+1] = -1/h
        K[j+1,j] = -1/h
    end
    return K
end

Load_vector = function (f,N)
    b = zeros(N)
    h = 1/(N+1)
    for j in 1:N
        b[j] = quadgk(x -> (f(x)*Basis_func(x,j,N)), (j-1)*h, j*h, (j+1)*h, rtol=1e-10)[1]
    end
    return b
end

Basis_func(x,j,N) = max(0, 1-abs(x*(N+1)-j))

Assemble_Solution = function (x)
    u = 0
    for j in 1:N
        u = u + c[j]*Basis_func(x,j,N)
    end
    return u
end

f(x) = sin(x)
N = 1000
K = Stiffness_Matrix(N)
b = Load_vector(f,N)
c = K \ b

plot_t = 0:0.01:1
plot(plot_t,Assemble_Solution.(plot_t),xlabel="x",label="FEM")