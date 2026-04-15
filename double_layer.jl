include("cal_energy.jl")

##### double_layer test #####
struct IPEPS_BULK 
    unitcell::unit_cell
    phy_dim::Int
    bond_dim::Int
    tensors
end    

function conv_double_env(ipepsbk::IPEPS_BULK,rand_env::IPEPS_ENV;iter = 60 ,tol = 1e-10)
    for i = 1:iter 
        _,sc1,_ = svd(rand_env.corner[2,2,1])
        rand_env = left_move(ipepsbk,rand_env)
        rand_env = right_move(ipepsbk,rand_env)
        rand_env = up_move(ipepsbk,rand_env)
        rand_env = down_move(ipepsbk,rand_env)
        _,sc,_ = svd(rand_env.corner[2,2,1])
        if norm(sc-sc1)<=tol && i >= 3
            break
            println("****converge_env finish, error**** ",norm(sc-sc1)," ****step**** ",i)
        end
    end
    return rand_env
end

function pre_step(bond_,chi_,g;iter = 60 ,tol = 1e-10)
      println("==== bond|",bond_,"==== chi|",chi_,"==== model|heisenberg")

      initial_ipeps = ini_IPEPS(bond=bond_);
      println("========ipeps prepared==========")
      rand_env = ini_bk_env(bond = bond_,chi = chi_)
      println("========bk rand env prepared==========")
      initial_bulk = GenWfbulk(initial_ipeps)
      conv_env = conv_double_env(initial_bulk,rand_env;iter = iter ,tol=tol)
      println("____==== env converged ====____")
      ham = GetH( -g,0 )
      return initial_ipeps,conv_env,ham
end

function GenWfbulk(ipeps::IPEPS)
    lenx = ipeps.unitcell.x 
    leny = ipeps.unitcell.y
    bond = ipeps.bond_dim
    phy = ipeps.phy_dim
    shapedim = bond^2
    tensor_data1 = Array{ Array{ Float64, 4 }, 2 }( undef, lenx, leny ) 
    tensor_data = Zygote.Buffer(tensor_data1)
    for i = 1:lenx,j=1:leny
        wftem = ipeps.tensors[i,j]
        @ein wfdb[1,2,3,4,5,6,7,8] := wftem[t,1,3,5,7]*wftem[t,2,4,6,8]
        wfdb = reshape(wfdb,(shapedim,shapedim,shapedim,shapedim))
        tensor_data[i,j] = copy(wfdb)
    end
    return IPEPS_BULK(ipeps.unitcell,phy,bond,tensor_data)
end


function optim_gs_double(tensors,conv_env::IPEPS_ENV,ham,iter_step,bond)
   
   tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 2, 2 ) 
   tensor_data = Zygote.Buffer(tensor_data1)
   tensor_data[1,1] = tensors[1,:,:,:,:,:]
   tensor_data[1,2] = tensors[2,:,:,:,:,:]
   tensor_data[2,1] = tensors[3,:,:,:,:,:]
   tensor_data[2,2] = tensors[4,:,:,:,:,:]
 
   unitcell= unit_cell(2,2)
   ipeps = IPEPS(unitcell,2,bond,tensor_data) 

   ipepsbulk = GenWfbulk(ipeps)
   
   for i = 1:iter_step
     conv_env = left_move(ipepsbulk,conv_env)
     conv_env = right_move(ipepsbulk,conv_env)
     conv_env = up_move(ipepsbulk,conv_env)
     conv_env = down_move(ipepsbulk,conv_env)
   end
   energy = calculate_bk_energy(ipeps,conv_env,ham)
   return energy
end



















































