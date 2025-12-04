using OptimalControl
using NLPModelsIpopt
using Plots

t1=[0.0,1.0]
t2=[cos(pi/6),-sin(pi/6)]
t3=[-cos(pi/6),-sin(pi/6)]
xf=[t1;t2;t3]
x0=[-0.5,-0.5,0.0,-0.2,0.8,-0.8]
x0_aug=[x0;0.0]
R_sq=0.2^2

@def ocp_aug begin
    tf∈R,variable
    t∈[0,tf],time
    x∈R⁷,state
    u∈R⁶,control
    x(0)==x0_aug
    x₁(tf)==xf[1]
    x₂(tf)==xf[2]
    x₃(tf)==xf[3]
    x₄(tf)==xf[4]
    x₅(tf)==xf[5]
    x₆(tf)==xf[6]
    tf≥0.1
    (x₁(t)-x₃(t))^2+(x₂(t)-x₄(t))^2≥R_sq
    (x₁(t)-x₅(t))^2+(x₂(t)-x₆(t))^2≥R_sq
    (x₃(t)-x₅(t))^2+(x₄(t)-x₆(t))^2≥R_sq
    ẋ(t)==[u₁(t),u₂(t),u₃(t),u₄(t),u₅(t),u₆(t),0.5*(u₁(t)^2+u₂(t)^2+u₃(t)^2+u₄(t)^2+u₅(t)^2+u₆(t)^2)]
    x₇(tf)+tf→min
end

println("Résolution...")
init_aug=(state=x0_aug,control=zeros(6),variable=1.0)
sol=solve(ocp_aug,init=init_aug,display=false)

tf_val=variable(sol)
ts=time_grid(sol)
x_func=state(sol)

println("Temps final optimal : ",tf_val)

x_val=[x_func(t) for t in ts]
mat=hcat(x_val...)'

p=plot(title="Morphogénèse (Succès)",aspect_ratio=:equal,legend=:outertopright)
plot!(p,mat[:,1],mat[:,2],label="Cellule 1",lw=3,c=:blue)
plot!(p,mat[:,3],mat[:,4],label="Cellule 2",lw=3,c=:red)
plot!(p,mat[:,5],mat[:,6],label="Cellule 3",lw=3,c=:green)
scatter!(p,[x0[1]],[x0[2]],m=:circle,c=:blue,alpha=0.3,label="Départ")
scatter!(p,[x0[3]],[x0[4]],m=:circle,c=:red,alpha=0.3,label="")
scatter!(p,[x0[5]],[x0[6]],m=:circle,c=:green,alpha=0.3,label="")
scatter!(p,[t1[1]],[t1[2]],m=:star,ms=10,c=:blue,label="Cible")
scatter!(p,[t2[1]],[t2[2]],m=:star,ms=10,c=:red,label="")
scatter!(p,[t3[1]],[t3[2]],m=:star,ms=10,c=:green,label="")
display(p)

energie_finale=mat[end,7]
println("Énergie totale : ",energie_finale)
