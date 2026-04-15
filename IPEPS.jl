
abstract type NTN_TENSOR end

struct unit_cell <:NTN_TENSOR 
    x::Int 
    y::Int
 end

mutable struct IPEPS <: NTN_TENSOR 
   unitcell ::unit_cell 
   phy_dim ::Int
   bond_dim ::Int
   tensors #::Array
end
 
mutable struct IPEPS_ENV <: NTN_TENSOR
   unitcell ::unit_cell
   chi ::Int 
   corner #::Array
   edge #::Array
end 

struct NTN_IPEPS <: NTN_TENSOR
    unitcell ::unit_cell 
    phy_dim ::Int
    bond_dim ::Int
    tensors #::Array
end

function Gen_NTNIPEPS(ipeps::IPEPS,id1,id2)
   x = 2*ipeps.unitcell.x 
   y = 2*ipeps.unitcell.y
   unitcell_NTN = unit_cell(x,y)
   bond = ipeps.bond_dim 
   phy = ipeps.phy_dim
   # println("bond",bond)
  # GEN id1 tensor 
   # id1st1 = zeros(Float64,phy,bond,bond,bond,phy,bond)
   # # id1st1 = Zygote.Buffer(id1st1tem)

   # for i1 = 1:bond,i2=1:bond,i3 = 1:bond,i4 = 1:bond,i5 = 1:phy,i6=1:phy
   #    if  i1 == i2 && i5 == i6 && i3 == i4 
   #          id1st1[i5,i1,i2,i3,i6,i4] = 1
   #    end   
   # end
   # id1 = Zygote.Buffer(zeros(phy*bond,bond,bond,phy*bond))
   # # id1re = zeros(phy,bond,bond,bond,phy,bond)
   # # id1re[:,:,:,:,:,:] = id1st1[:,:,:,:,:,:]

   # id1re1 = reshape(id1st1,(phy*bond,bond,bond,phy*bond))
   # # GEN id2 tensor 
   # id1[:,:,:,:] = id1re1[:,:,:,:]

   # id2 = zeros(Float64,bond,bond,bond,bond)
   # # id2 = Zygote.Buffer(id2tem)
   # for i1 = 1:bond,i2=1:bond,i3=1:bond,i4=1:bond
   #     if i1 == i2 && i3 ==i4 
   #        id2[i1,i2,i3,i4] = 1
   #     end
   # end
   tensor_data1 = Array{ Array{ Float64, 4 }, 2 }( undef, 2, y ) 
   tensor_data = Zygote.Buffer(tensor_data1)
   # tensor_data = tensor_data1[:,:]
   for i = 1:2,j=1:y 
      if i%2 == 0
        if j%2 == 0
           tensortem = ipeps.tensors[ceil(Int,i/2),ceil(Int,j/2)]
           data = permutedims(tensortem,(2,3,1,4,5))
           
           tensor_data[i,j] = reshape(data,(bond,bond,phy*bond,bond))
        else
           tensor_data[i,j] =  copy(id2)
        end
      else 
        if j%2 == 0
            tensor_data[i,j] =  copy(id1)
        else
            tensortem = ipeps.tensors[ceil(Int,i/2),ceil(Int,j/2)]
            
            data = permutedims(tensortem,(2,1,3,4,5))
            # println(size(data))
            # println(bond,phy)
            tensor_data[i,j] = reshape(data,(bond,phy*bond,bond,bond))
        end
      end
   end
   return NTN_IPEPS(unitcell_NTN,phy,bond,tensor_data)
end



function ini_IPEPS(loadname,data_select;x = 2,y = 2,bond = 5,phy = 2)
    unitcell = unit_cell(x,y)
    tensor_data = Array{ Array{ Float64, 5 }, 2 }( undef, 1, y )  
    
    if data_select == "rand_data"
      for i = 1:1,j=1:y
         tensor_data[i,j] = rand(Float64,phy,bond,bond,bond,bond)
      end
    else
      wf_dict = load(loadname)
      wfc = wf_dict["wf"]
      for i = 1:1,j=1:y
         println(size(wfc))
         tensor_data[i,j] = wfc[(i-1)*y+j,:,:,:,:,:]
      end
    end
    return IPEPS(unitcell,phy,bond,tensor_data)
end

function ini_env(;x = 4,y= 4,chi = 20, bond = 5,phy = 2)
    unitcell = unit_cell(x,y)
    corner_data = Array{ Array{ Float64, 2 }, 3 }( undef, 2,y, 4 ) 
    edge_data = Array{ Array{ Float64, 3 }, 3 }( undef, 2,y, 4 )
    
    for i = 1:2, j = 1:y,k=1:4 
        tem = randn(Float64,chi,chi)
        corner_data[i,j,k] = tem#randn(Float64,chi,chi)
        if i%2 == 0 
           if j%2 ==0
              if k == 3 
                 edge_data[i,j,k] = rand(Float64,phy*bond,chi,chi)
              else 
                edge_data[i,j,k] = rand(Float64,bond,chi,chi)
              end
           else 
            edge_data[i,j,k] = rand(Float64,bond,chi,chi)
           end
        else
           if j%2 == 0
              if k ==1 || k ==4
                edge_data[i,j,k] = rand(Float64,phy*bond,chi,chi)
              else
                edge_data[i,j,k] = rand(Float64,bond,chi,chi)
              end
           else
              if k ==2 
                edge_data[i,j,k] = rand(Float64,phy*bond,chi,chi)
              else 
                edge_data[i,j,k] = rand(Float64,bond,chi,chi)
              end
           end
        end
    end    
    return IPEPS_ENV(unitcell,chi,corner_data,edge_data)
end


# a = ini_IPEPS();
# b = ini_env();
# c = Gen_NTNIPEPS(a);
# println(1)

function ini_bk_env(;x=2,y=4,chi=20,bond = 3,phy = 2)
   unitcell = unit_cell(x,y)
   corner_data = Array{ Array{ Float64, 2 }, 3 }( undef, x,y, 4 ) 
   edge_data = Array{ Array{ Float64, 3 }, 3 }( undef, x,y, 4 )
   for i =1:x,j=1:y,k=1:4
       corner_data[i,j,k] = rand(chi,chi) 
       edge_data[i,j,k] = randn(bond^2,chi,chi)
   end
   return IPEPS_ENV(unitcell,chi,corner_data,edge_data)
end












