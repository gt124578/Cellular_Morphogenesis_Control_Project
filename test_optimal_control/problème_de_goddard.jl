using OptimalControl
using NLPModelsIpopt
using Plots

ocp=@def begin
    tf ∈ R,variable
    t ∈ [0,tf],time
    x ∈ R³,state
    u ∈ R,control
    x(0)==[1,0,1]
    x₃(tf)==0.6
    0≤u(t)≤1
    x₁(t)≥1.0
    ẋ(t)==[x₂(t),
            (3.5*u(t)-310.0*x₂(t)^2*exp(-500.0*(x₁(t)-1.0)))/x₃(t)-1.0/x₁(t)^2,
            -3.5*u(t)]
    x₁(tf)→max
end

println("Résolution en cours...")
init=(state=[1.01,0.05,0.8],control=0.5,variable=0.2)
sol=solve(ocp,grid_size=100,init=init,display=false)
println("Altitude atteinte : ",objective(sol))
plot(sol,size=(800,600))
