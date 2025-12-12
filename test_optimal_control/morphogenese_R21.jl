using OptimalControl
using NLPModelsIpopt
using Plots

Nb_Cellules=10 #nombre de cellules
dim_x=2*Nb_Cellules+1 #états : position+énergie
dim_u=2*Nb_Cellules #dimension du contrôle

#rayon minimal pour éviter les collisions entre les cellules 
Rayon_sq=0.15^2
state_names=join(["x$i" for i in 1:dim_x],", ")

control_names=join(["u$i" for i in 1:dim_u],", ")

#construction (dynamique) de optimalControl
code_str="""
@def ocp_generated begin
    tf ∈ R,variable
    t ∈ [0,tf],time
    x=($state_names) ∈ R^$dim_x,state
    u=($control_names) ∈ R^$dim_u,control
"""

#Condition initiales
for i in 1:Nb_Cellules
    val_x=-1.0+2.0*(i-1)/(Nb_Cellules-1)
    val_y=0.0
    global code_str*="    x$(2*(i-1)+1)(0)==$val_x\n"
    global code_str*="    x$(2*i)(0)==$val_y\n"
end

code_str*="    x$dim_x(0)==0.0\n" #énergie initiale (nulle)

#cible (disposées en cercle)
theta=range(0,2pi,length=Nb_Cellules+1)[1:end-1]

for i in 1:Nb_Cellules
    val_x=cos(theta[i])
    val_y=sin(theta[i])
    global code_str*="    x$(2*(i-1)+1)(tf)==$val_x\n"
    global code_str*="    x$(2*i)(tf)==$val_y\n"
end

code_str*="    tf≥0.1\n" #borne inf au temps final

#contrainte anti-collision
for i in 1:Nb_Cellules
    for j in (i+1):Nb_Cellules
        ix=2*(i-1)+1
        iy=2*i
        jx=2*(j-1)+1
        jy=2*j
        global code_str*="    (x$ix(t)-x$jx(t))^2+(x$iy(t)-x$jy(t))^2≥$Rayon_sq\n"
    end
end

#dynamique
dyn_list=["u$i(t)" for i in 1:dim_u]
energy_terms=join(["u$i(t)^2" for i in 1:dim_u]," + ")
push!(dyn_list,"0.5*($energy_terms)")
dyn_str=join(dyn_list,",\n              ")
code_str*="""
    ẋ(t)==[$dyn_str]
    x$dim_x(tf)+tf→min  #coût (énergie+durée)
end
"""

println("Génération du modèle pour $Nb_Cellules cellules...")
eval(Meta.parse(code_str))

#transcription directe+Ipopt (solver non linéaire)
println("Résolution en cours (Dimension $dim_x, 45 contraintes de collision)...")
init_val=(state=zeros(dim_x),control=zeros(dim_u),variable=1.0)
sol=solve(ocp_generated,init=init_val,display=false)


tf_opt=variable(sol)[1]

println("Terminé en T = $tf_opt")

#reconstruction de la trajectoire 
ts=time_grid(sol)
x_func=state(sol)
x_val=[x_func(t) for t in ts]
mat=hcat(x_val...)'

#pour la creation du .gif
frames = 1:size(mat,1)
anim = @animate for k in frames
    p = plot(title="Morphogénèse ($Nb_Cellules Cellules)",aspect_ratio=:equal,legend=false)
    for i in 1:Nb_Cellules
        idx_x = 2*(i-1)+1
        idx_y = 2*i
        plot!(p,mat[1:k,idx_x],mat[1:k,idx_y],lw=2,c=colors[i])
        scatter!(p,[mat[k,idx_x]],[mat[k,idx_y]],m=:circle,c=colors[i])
    end
end

gif(anim,"morphogenese.gif",fps=20)

#visualisation
p=plot(title="Morphogénèse ($Nb_Cellules Cellules)",aspect_ratio=:equal,legend=false)
colors=range(HSL(0,1,0.5),HSL(330,1,0.5),length=Nb_Cellules)



for i in 1:Nb_Cellules
    idx_x=2*(i-1)+1
    idx_y=2*i
    plot!(p,mat[:,idx_x],mat[:,idx_y],lw=2,c=colors[i]) #trajectoire de i
    x_start=-1.0+2.0*(i-1)/(Nb_Cellules-1)
    scatter!(p,[x_start],[0.0],m=:circle,c=colors[i],alpha=0.4) #position initiale
    x_target=cos(theta[i])
    y_target=sin(theta[i])
    scatter!(p,[x_target],[y_target],m=:star,ms=8,c=colors[i]) #cible
end

#affichage
display(p)

println("Énergie totale : ",mat[end,end])

