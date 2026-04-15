include("ctmrg.jl")
function GetH( J,MarshallSign )
    XX = kron( Sx, Sx )
    YY = kron( Sy, Sy )
    ZZ = kron( Sz, Sz )
    Ham = - J*(XX - YY + ZZ ) #XX - YY + 
    if MarshallSign ==1
       Ham[2,3] = -Ham[2,3]
       Ham[3,2] = -Ham[3,2]
    end
    return Ham
end    





function calculate_energy(ipeps_ntn::NTN_IPEPS,env_ce::IPEPS_ENV,ham::Array)
    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y
    ### vertical energy ###
    # println(m)
    # println(n)
    
    bond = ipeps_ntn.bond_dim
    phy = ipeps_ntn.phy_dim

    energy = 0

    for i =1:2:2,j=1:2:n 
        # println("aindex","i",i,"j",j)

        a1 = ipeps_ntn.tensors[i,j]
        a1d = ipeps_ntn.tensors[i+1,j+1]
        a2 = ipeps_ntn.tensors[i,(j+2)%n]
        a2d = ipeps_ntn.tensors[i+1,(j+2)%n+1]

        # println("adindex","i",i+1,"j",j+1)

        # println("a2index","i",i,"j",(j+2)%n)

        # println("a2dindex","i",i+1,"j",(j+2)%n+1)

        c1 = env_ce.corner[i,j,1]
        c2 = env_ce.corner[i,(j+2)%n+1,2] 
        c3 = env_ce.corner[i+1,j,3] 
        c4 = env_ce.corner[i+1,(j+2)%n+1,4] 

        e1a = env_ce.edge[i,j,1]
        e1b = env_ce.edge[i+1,j,1]

        e2a = env_ce.edge[i,(j+2)%n+1,2] 
        e2b = env_ce.edge[i+1,(j+2)%n+1,2]

        e3a = env_ce.edge[i,j,3] 
        e3b = env_ce.edge[i,j+1,3]
        e3c = env_ce.edge[i,(j+2)%n,3]
        e3d = env_ce.edge[i,(j+2)%n + 1,3]


        e4a = env_ce.edge[i+1,j,4]
        e4b = env_ce.edge[i+1,j+1,4]
        e4c = env_ce.edge[i+1,(j+2)%n,4]
        e4d = env_ce.edge[i+1,(j+2)%n + 1,4]

        # contract
        a1 = reshape(a1,(bond,phy,bond,bond,bond))
        a1d = reshape(a1d,(bond,bond,phy,bond,bond))
        a2 = reshape(a2,(bond,phy,bond,bond,bond))
        a2d = reshape(a2d,(bond,bond,phy,bond,bond))

        a1 = permutedims(a1,(2,1,3,4,5))
        a1d = permutedims(a1d,(3,1,2,4,5))
        a2 = permutedims(a2,(2,1,3,4,5))
        a2d = permutedims(a2d,(3,1,2,4,5))

        @ein leftop[1,2,3,4,5] := a1[5,m,1,n,3]*e3a[n,p,2]*c1[p,q]*e1a[m,q,4]
        @ein left1[1,2,3,4,5] := leftop[1,2,m,n,5]*e1b[3,n,t]*c3[t,p]*e4a[m,p,4]
        @ein left[1,2,3,4,5,6] := left1[1,t,m,n,5]*a1d[6,m,3,p,q]*e3b[p,t,2]*e4b[q,n,4]

        @ein rightdown[1,2,3,4,5] := a2d[5,1,m,3,n]*e2b[m,4,p]*e4d[n,2,q]*c4[p,q]
        @ein right1[1,2,3,4,5] :=  rightdown[1,2,m,n,5]*e3d[m,4,p]*c2[p,q]*e2a[3,q,n]
        @ein right[1,2,3,4,5,6] := right1[1,u,m,n,6]*a2[5,3,m,t,p]*e3c[t,4,n]*e4c[p,2,u]
        
        @ein rho[1,2,3,4] := left[m,n,p,q,1,3]*right[p,q,m,n,2,4]
        rho = reshape(rho,(phy*phy,phy*phy))
        # println("rho1",rho)
        etm = dot(rho,ham)/tr(rho)         
        energy = energy + etm
    end
    
    for i = 1:2:n,j=1:2:2
       # println("aindex","i",i,"j",j)
        # println("aindex","i",j,"j",i)
        idx1,idy1 = map_neel_pattern[j,i]
        idx2,idy2 = map_neel_pattern[j+1,i+1]
        idx3,idy3 = map_neel_pattern[(j+2)%m,i]
        idx4,idy4 = map_neel_pattern[(j+2)%m+1,i+1]

        a1 = ipeps_ntn.tensors[idx1,idy1]
        a1d = ipeps_ntn.tensors[idx2,idy2]
        a2 = ipeps_ntn.tensors[idx3,idy3]
        a2d = ipeps_ntn.tensors[idx4,idy4]


        

        # println("adindex","i",j+1,"j",i+1)

        # println("a2index","i",(j+2)%m,"j",i)

        # println("a2dindex","i",(j+2)%m+1,"j",i+1)
        idx1,idy1 = map_neel_pattern[j,i]
        idx2,idy2 = map_neel_pattern[j,i+1]
        idx3,idy3 = map_neel_pattern[(j+2)%m+1,i]
        idx4,idy4 = map_neel_pattern[(j+2)%m+1,i+1]

        c1 = env_ce.corner[idx1,idy1,1]
        c2 = env_ce.corner[idx2,idy2,2]
        c3 = env_ce.corner[idx3,idy3,3]
        c4 = env_ce.corner[idx4,idy4,4]





        e3a = env_ce.edge[idx1,idy1,3]
        e3b = env_ce.edge[idx2,idy2,3]
        
        e4a = env_ce.edge[idx3,idy3,4]
        e4b = env_ce.edge[idx4,idy4,4]



        idx1,idy1 = map_neel_pattern[j,i]
        idx2,idy2 = map_neel_pattern[j+1,i]
        idx3,idy3 = map_neel_pattern[(j+2)%m,i]
        idx4,idy4 = map_neel_pattern[(j+2)%m+1,i]


        e1a = env_ce.edge[idx1,idy1,1]
        e1b = env_ce.edge[idx2,idy2,1]
        e1c = env_ce.edge[idx3,idy3,1]
        e1d = env_ce.edge[idx4,idy4,1]


        idx1,idy1 = map_neel_pattern[j,i+1]
        idx2,idy2 = map_neel_pattern[j+1,i+1]
        idx3,idy3 = map_neel_pattern[(j+2)%m,i+1]
        idx4,idy4 = map_neel_pattern[(j+2)%m+1,i+1]


        e2a = env_ce.edge[idx1,idy1,2]
        e2b = env_ce.edge[idx2,idy2,2]
        e2c = env_ce.edge[idx3,idy3,2]
        e2d = env_ce.edge[idx4,idy4,2]

        a1 = reshape(a1,(bond,phy,bond,bond,bond))
        a1d = reshape(a1d,(bond,bond,phy,bond,bond))
        a2 = reshape(a2,(bond,phy,bond,bond,bond))
        a2d = reshape(a2d,(bond,bond,phy,bond,bond))

        a1 = permutedims(a1,(2,1,3,4,5))
        a1d = permutedims(a1d,(3,1,2,4,5))
        a2 = permutedims(a2,(2,1,3,4,5))
        a2d = permutedims(a2d,(3,1,2,4,5))

        @ein topleft[1,2,3,4,5] := a1[5,q,3,p,1]*e1a[q,n,2]*c1[m,n]*e3a[p,m,4]
        @ein top1[1,2,3,4,5] := topleft[1,2,m,n,5]*e3b[3,n,p]*c2[p,q]*e2a[m,q,4]
        @ein top[1,2,3,4,5,6] :=top1[1,m,n,p,5]*a1d[6,u,t,n,3]*e2b[t,p,4]*e1b[u,m,2]

        @ein downright[1,2,3,4,5] := a2d[5,3,m,1,n]*e2d[m,2,p]*c4[p,q]*e4b[n,4,q]
        @ein down1[1,2,3,4,5] := downright[1,2,p,q,5]*e1d[p,4,u]*e4a[3,t,q]*c3[u,t]
        @ein down[1,2,3,4,5,6] := down1[1,m,n,p,6]*a2[5,u,t,3,n]*e2c[t,2,m]*e1c[u,4,p]

        @ein rho[1,2,3,4] := top[m,n,p,q,1,3]*down[p,q,m,n,2,4]
        rho = reshape(rho,(phy*phy,phy*phy))
        # println("rho2",rho)
        etm = dot(rho,ham)/tr(rho)         
        energy = energy + etm

    end
    return energy*8/(m*n)
end



### hori term 
function calculate_bk_energy(ipeps,conv_env,ham)
    m = ipeps.unitcell.x
    n = ipeps.unitcell.y

    phy = ipeps.phy_dim
    bond = ipeps.bond_dim
    chi_ = conv_env.chi
    energy = 0
    for i = 1:m,j=1:n 
        # println(i,j," ",i,j%n+1)
        c1 = conv_env.corner[i,j,1]
        c2 = conv_env.corner[i,j%n+1,2]
        c3 = conv_env.corner[i,j,3]
        c4 = conv_env.corner[i,j%n+1,4]
      
        e1 = conv_env.edge[i,j,1]
        e2 = conv_env.edge[i,j%n+1,2]

        e3a = conv_env.edge[i,j,3]
        e3b = conv_env.edge[i,j%n+1,3]

        e4a = conv_env.edge[i,j,4]
        e4b = conv_env.edge[i,j%n+1,4]

        a1 = ipeps.tensors[i,j]
        a2 = ipeps.tensors[i,j%n+1]
        
        e1 = reshape(e1,(bond,bond,chi_,chi_))
        e2 = reshape(e2,(bond,bond,chi_,chi_))

        e3a = reshape(e3a,(bond,bond,chi_,chi_))
        e3b = reshape(e3b,(bond,bond,chi_,chi_))

        e4a = reshape(e4a,(bond,bond,chi_,chi_))
        e4b = reshape(e4b,(bond,bond,chi_,chi_))

        @ein lef_topts[1,2,3,4,5,6] :=e1[m,n,p,q]*a1[5,m,2,m1,m2]*a1[6,n,3,n1,n2]*c1[p1,p]*c3[q,p2]*e3a[m1,n1,p1,1]*e4a[m2,n2,p2,4]
        @ein rig_dowts[1,2,3,4,5,6] :=e2[m,n,p,q]*a2[5,2,m,m1,m2]*a2[6,3,n,n1,n2]*c2[q1,p]*c4[q,q2]*e3b[m1,n1,1,q1]*e4b[m2,n2,4,q2]
        
        @ein Rho[5,6,7,8] := lef_topts[1,2,3,4,5,7]*rig_dowts[1,2,3,4,6,8]
        Rho = reshape(Rho,(4,4))
        etm = dot( Rho, ham ) / tr( Rho )
        energy += etm
    end

    for i = 1:m,j=1:n 

        # println(i,j)
        # println(i%m+1,j)


        c1 = conv_env.corner[i,j,1]
        c2 = conv_env.corner[i,j,2]
        c3 = conv_env.corner[i%m+1,j,3]
        c4 = conv_env.corner[i%m+1,j,4]

        e3 = conv_env.edge[i,j,3]
        e4 = conv_env.edge[i%m+1,j,4]
 
        e1a = conv_env.edge[i,j,1]
        e1b = conv_env.edge[i%m+1,j,1]
        
        e2a = conv_env.edge[i,j,2]
        e2b = conv_env.edge[i%m+1,j,2]

        e3 = reshape(e3,(bond,bond,chi_,chi_))
        e4 = reshape(e4,(bond,bond,chi_,chi_))

        e1a = reshape(e1a,(bond,bond,chi_,chi_))
        e1b = reshape(e1b,(bond,bond,chi_,chi_))

        e2a = reshape(e2a,(bond,bond,chi_,chi_))
        e2b = reshape(e2b,(bond,bond,chi_,chi_))

        a1 = ipeps.tensors[i,j]
        a2 = ipeps.tensors[i%m+1,j]

        @ein lef_topts[1,2,3,4,5,6] := e3[m,n,p,q]*a1[5,m1,m2,m,2]*a1[6,n1,n2,n,3]*c1[p,p1]*c2[q,p2]*e1a[m1,n1,p1,1]*e2a[m2,n2,p2,4]
        @ein rig_dowts[1,2,3,4,5,6] := e4[m,n,p,q]*a2[5,m2,m1,2,m]*a2[6,n2,n1,3,n]*c3[p2,p]*c4[p1,q]*e1b[m2,n2,1,p2]*e2b[m1,n1,4,p1]

        @ein Rho[5,6,7,8] := lef_topts[1,2,3,4,5,7]*rig_dowts[1,2,3,4,6,8]
        Rho = reshape(Rho,(4,4))
        etm = dot( Rho, ham ) / tr( Rho )
        energy += etm
    end
    return energy/(m*n)
end









