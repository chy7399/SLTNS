
function Genconst(phy,bond)
    id1st1 = zeros(Float64,phy,bond,bond,bond,phy,bond)
    # id1st1 = Zygote.Buffer(id1st1tem)
 
    for i1 = 1:bond,i2=1:bond,i3 = 1:bond,i4 = 1:bond,i5 = 1:phy,i6=1:phy
       if  i1 == i2 && i5 == i6 && i3 == i4 
             id1st1[i5,i1,i2,i3,i6,i4] = 1
       end   
    end
    id1 = reshape(id1st1,(phy*bond,bond,bond,phy*bond))

    id2 = zeros(Float64,bond,bond,bond,bond)
    for i1 = 1:bond,i2=1:bond,i3=1:bond,i4=1:bond
        if i1 == i2 && i3 ==i4 
           id2[i1,i2,i3,i4] = 1
        end
    end
    return id1,id2
end
# id1t,id2t = Genconst(2,2)
# const id1 = copy(id1t)
# const id2 = copy(id2t)

function converge_env(initial_ntnipeps,rand_env::IPEPS_ENV,iter::Int,tol::Float64,bond,id1,id2)
    println("****toleration set**** ",tol)
    for i = 1:iter 
        println("step",i)
        _,sc1,_ = svd(rand_env.corner[1,1,1])
        # println("edge1 ",rand_env.edge[1,1,1][1,1,1])
        rand_env = left_move(initial_ntnipeps,rand_env,bond,id1,id2)
        # println("edge2 ",rand_env.edge[1,1,1][1,1,1])
        rand_env = right_move(initial_ntnipeps,rand_env,bond,id1,id2)
        # println("edge3 ",rand_env.edge[1,1,1][1,1,1])
        rand_env = up_move(initial_ntnipeps,rand_env,bond,id1,id2)
        # println("edge4 ",rand_env.edge[1,1,1][1,1,1])
        rand_env = down_move(initial_ntnipeps,rand_env,bond,id1,id2)
        # println("edge5 ",rand_env.edge[1,1,1][1,1,1])
        _,sc,_ = svd(rand_env.corner[1,1,1])
        # println(norm(sc-sc1))
        if norm(sc-sc1)<=tol && i >= 3
            println("****converge_env finish, error**** ",norm(sc-sc1)," ****step**** ",i)
            break    
        end
        if i == iter
            println("=========dont get==========")
            println(norm(sc-sc1))
        end
    end
    return rand_env
end




function pre_step(loadname,data_select,bond_,chi_,g,id1,id2;iter = 200, tol= 1e-10)
    


    println("|bond dimension| ",bond_,"|boundary dimension| ",chi_,"|ham J| ",g,"|converge_env iter| ",iter,"|converge_env tol|",tol)
    println("*******************************")
    initial_ipeps = ini_IPEPS(loadname,data_select,bond=bond_);
    println("initial_ipeps complished")
    println("*******************************")
    rand_env = ini_env(bond=bond_,chi=chi_)
    println("randenv complished")
    println("*******************************")
    initial_ntnipeps = Gen_NTNIPEPS(initial_ipeps,id1,id2)
    println("ntn complished")
    println("*******************************")

    start = zeros(2,2,bond_,bond_,bond_,bond_)
    start[1,:,:,:,:,:] = initial_ipeps.tensors[1,1]
    start[2,:,:,:,:,:] = initial_ipeps.tensors[1,2]
    # start[3,:,:,:,:,:] = initial_ipeps.tensors[2,1]
    # start[4,:,:,:,:,:] = initial_ipeps.tensors[2,2]

    conv_env = converge_env(start,rand_env,60,tol,bond_,id1,id2)
    ham = GetH( -g,0 )

    return initial_ipeps,initial_ntnipeps,conv_env,ham
end


function optim_gs1(tensors,conv_env,ham,grad_ctmrg_step,bond,id1,id2)
   # warm up with no grad  

 #  println(conv_env.edge[1,1,1][1,1,1])
   tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 1, 2 ) 
   tensor_data = Zygote.Buffer(tensor_data1)
   tensor_data[1,1] = tensors[1,:,:,:,:,:]
   tensor_data[1,2] = tensors[2,:,:,:,:,:]
#   tensor_data[2,1] = tensors[3,:,:,:,:,:]
#   tensor_data[2,2] = tensors[4,:,:,:,:,:]

   unitcell= unit_cell(2,2)
   initial_ipeps = IPEPS(unitcell,2,bond,tensor_data) 
   initial_ntnipeps = Gen_NTNIPEPS(initial_ipeps,id1,id2)

#    Zygote.ignore() do
#     # initial_ipeps 
#     # initial_ntnipeps = Gen_NTNIPEPS(initial_ipeps)
#     conv_env = converge_env(initial_ntnipeps,conv_env,200,1e-8)
#    end

   # ctmrg with grad
   for i = 1:grad_ctmrg_step 
   
     conv_env = left_move(tensors,conv_env,bond,id1,id2)  
  
     conv_env = right_move(tensors,conv_env,bond,id1,id2)

     conv_env = up_move(tensors,conv_env,bond,id1,id2)
 
     conv_env = down_move(tensors,conv_env,bond,id1,id2)
 
   end

   energy = calculate_energy(initial_ntnipeps,conv_env,ham)   
   #println("test",energy)
   return energy   #calculate_energy(initial_ntnipeps,conv_env,ham)
end


function optim_gs(tensors,conv_env,ham,iter_ctmrg_step,bond,id1,id2,grad_ctmrg_step)
    # warm up with no grad  
 
  #  println(conv_env.edge[1,1,1][1,1,1])
    tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 1, 2 ) 
    tensor_data = Zygote.Buffer(tensor_data1)
    tensor_data[1,1] = tensors[1,:,:,:,:,:]
    tensor_data[1,2] = tensors[2,:,:,:,:,:]
    # tensor_data[2,1] = tensors[3,:,:,:,:,:]
    # tensor_data[2,2] = tensors[4,:,:,:,:,:]
 
    unitcell= unit_cell(2,2)
    initial_ipeps = IPEPS(unitcell,2,bond,tensor_data) 
    initial_ntnipeps = Gen_NTNIPEPS(initial_ipeps,id1,id2)
 
 #    Zygote.ignore() do
 #     # initial_ipeps 
 #     # initial_ntnipeps = Gen_NTNIPEPS(initial_ipeps)
 #     conv_env = converge_env(initial_ntnipeps,conv_env,200,1e-8)
 #    end
 
    # ctmrg with grad
    for i = 1:iter_ctmrg_step 
  #    println(conv_env.edge[1,1,1][1,1,1])
      conv_env = left_move(tensors,conv_env,bond,id1,id2)
  #    println(conv_env.edge[1,1,1][1,1,1])
      conv_env = right_move(tensors,conv_env,bond,id1,id2)
  #    println(conv_env.edge[1,1,1][1,1,1])
      conv_env = up_move(tensors,conv_env,bond,id1,id2)
  #    println(conv_env.edge[1,1,1][1,1,1])
      conv_env = down_move(tensors,conv_env,bond,id1,id2)
  #    println(conv_env.edge[1,1,1][1,1,1])
    end
 
     energy = calculate_energy(initial_ntnipeps,conv_env,ham)
     println("test",energy)
    return energy   #calculate_energy(initial_ntnipeps,conv_env,ham)
 end
