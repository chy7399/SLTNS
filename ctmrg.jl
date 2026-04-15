include("IPEPS.jl")
function Genindex_ctmrg(m,n,_index_,choose_flag)
    index1 = 0
    index2 = 0
    index3 = 0
    index4 = 0
    if choose_flag == "down"
        index1 = (_index_ - n) <= 0 ? (_index_ - n) + m*n : (_index_ - n)
        index2 = index1%n + n* Int(ceil(index1/n) - 1) + 1
        index3 = _index_
        index4 = (index3)%n + n* Int(ceil((index3)/n) - 1) + 1
    elseif choose_flag == "left"
        index1 = _index_
        index2 = index1%n + n* Int(ceil(index1/n) - 1) + 1
        index3 = (_index_ + n) <= m*n ? (_index_ + n) : (_index_ + n)- m*n
        index4 = (index3)%n + n* Int(ceil((index3)/n) - 1) + 1
    elseif choose_flag == "right"
        index4 = _index_
        index2 = _index_ - n <=0 ? _index_ - n + m*n : _index_ - n
        index1 = (index2 - 1)%n ==0 ? index2 + n - 1 : index2 - 1
        index3 = (index4 - 1)%n == 0 ? index4 - 1 + n : index4 - 1
    elseif choose_flag == "up"
        index2 = _index_
        index1 = (index2 - 1)% n == 0 ? index2 - 1 + n : index2 - 1
        index4 = (_index_ + n) <= m*n ? (_index_ + n) : (_index_ + n)- m*n
        index3 = (index4 - 1)% n == 0  ? index4 - 1 + n : index4 - 1
    end     
    return index1,index2,index3,index4
end


function Genxy(idx1,idx2,idx3,idx4,y)
  idx1x = 0
  idx1y = 0
  idx2x = 0
  idx2y = 0
  idx3x = 0
  idx3y = 0
  idx4x = 0
  idx4y = 0

  idx1x = Int(ceil(idx1/y))
  idx1y = idx1-y*(idx1x-1) 
  idx2x = Int(ceil(idx2/y)) 
  idx2y = idx2-y*(idx2x-1) 
  idx3x = Int(ceil(idx3/y)) 
  idx3y = idx3-y*(idx3x-1) 
  idx4x = Int(ceil(idx4/y)) 
  idx4y = idx4-y*(idx4x-1) 
  return idx1x,idx1y,idx2x,idx2y,idx3x,idx3y,idx4x,idx4y
end




function GenIso_down(_index_,ipeps_ntn,env_ce)
    # bond = ipeps_ntn.bond_dim
    # phy = ipeps_ntn.phy_dim
    chi = env_ce.chi
  
    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y

   
    idx3x,idx3y = _index_[1],_index_[2] 
    idx4x,idx4y = idx3x,idx3y%4+1
    idx1x,idx1y = idx3x-1 == 0 ? 4 : idx3x-1,idx3y
    idx2x,idx2y = idx1x,idx3y%4+1


    idx1x,idx1y = map_neel_pattern[idx1x,idx1y]
    idx2x,idx2y = map_neel_pattern[idx2x,idx2y]
    idx3x,idx3y = map_neel_pattern[idx3x,idx3y]
    idx4x,idx4y = map_neel_pattern[idx4x,idx4y]

    c1 = copy(env_ce.corner[idx1x,idx1y,1])
    c2 = copy(env_ce.corner[idx2x,idx2y,2])
    c3 = copy(env_ce.corner[idx3x,idx3y,3]) 
    c4 = copy(env_ce.corner[idx4x,idx4y,4])

    e1a = copy(env_ce.edge[idx1x,idx1y,1])
    e1b = copy(env_ce.edge[idx3x,idx3y,1])

    e2a = copy(env_ce.edge[idx2x,idx2y,2])
    e2b = copy(env_ce.edge[idx4x,idx4y,2])

    e3a = copy(env_ce.edge[idx1x,idx1y,3])
    e3b = copy(env_ce.edge[idx2x,idx2y,3])

    e4a = copy(env_ce.edge[idx3x,idx3y,4])
    e4b = copy(env_ce.edge[idx4x,idx4y,4])

    a1 = copy(ipeps_ntn.tensors[idx1x,idx1y])
    a2 = copy(ipeps_ntn.tensors[idx2x,idx2y])
    a3 = copy(ipeps_ntn.tensors[idx3x,idx3y])
    a4 = copy(ipeps_ntn.tensors[idx4x,idx4y])

    dim1 = chi
    dim2 = size(a1)[2]
    dim3 = size(a3)[2]


    x_tensor = zeros(Float64,dim2,dim1,dim3,dim1)
    y_tensor = zeros(Float64,dim2,dim1,dim3,dim1)

    uptensor = ein"myni,npu,pq,mqo -> yuio"(a1,e3a,c1,e1a)
    downtensor = ein"myin,ntu,pt,mop -> yuio"(a3,e4a,c3,e1b) 
    x_tensor = ein"yumn,iomn -> yuio"(uptensor,downtensor)
    
    uptensor = ein"ymni,nup,mqo,pq- > yuio"(a2,e3b,e2a,c2)
    downtensor = ein"ymin,mop,nuq,pq -> yuio"(a4,e2b,e4b,c4)
    y_tensor = ein"yumn,iomn -> yuio"(uptensor,downtensor)  

    x_matrix = reshape(x_tensor,(dim2*dim1,dim3*dim1))
    y_matrix = reshape(y_tensor,(dim2*dim1,dim3*dim1))
    
    x_matrix = x_matrix/maximum(abs.(x_matrix))
    y_matrix = y_matrix/maximum(abs.(y_matrix))

    x_matrix = x_matrix.*100
    y_matrix = y_matrix.*100

    L_matrix = x_matrix
    R_matrix = y_matrix
    R_matrix = R_matrix'

    u,s,v = MySVD.mysvd1(L_matrix*R_matrix)
    # println("sizes",size(s)[1])
    # println("truncation error",sum(s[chi:size(s)[1]])/sum(s))
    # println("trunction error",sum(s[chi:size(s)[1]])/sum(s))
    # println(s)
    s1 = s[1:chi]
    u1 = u'[1:chi,:] 
    v1 = v[:,1:chi] 
    s1_inv = sqrt.(1.0./s1)
    
    pd_matrix = R_matrix * v1 * Diagonal(s1_inv)
    qd_matrix = Diagonal(s1_inv) * u1 * L_matrix
    
    qd_matrix = qd_matrix'

    pd_tensor = reshape(pd_matrix,(dim3,chi,chi))
    qd_tensor = reshape(qd_matrix,(dim3,chi,chi))

   return pd_tensor,qd_tensor#,s1
end

function GenIso_up(_index_,ipeps_ntn,env_ce::IPEPS_ENV)
    chi = env_ce.chi
  
    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y


    idx2x,idx2y = _index_[1],_index_[2] 
    idx1x,idx1y = idx2x,idx2y-1 == 0 ? 4 : idx2y-1
    idx3x,idx3y = idx1x%4+1,idx1y
    idx4x,idx4y = idx3x,idx3y%4+1



    idx1x,idx1y = map_neel_pattern[idx1x,idx1y]
    idx2x,idx2y = map_neel_pattern[idx2x,idx2y]
    idx3x,idx3y = map_neel_pattern[idx3x,idx3y]
    idx4x,idx4y = map_neel_pattern[idx4x,idx4y]




    c1 = copy(env_ce.corner[idx1x,idx1y,1])
    c2 = copy(env_ce.corner[idx2x,idx2y,2])
    c3 = copy(env_ce.corner[idx3x,idx3y,3])
    c4 = copy(env_ce.corner[idx4x,idx4y,4])

    e1a = copy(env_ce.edge[idx1x,idx1y,1])
    e1b = copy(env_ce.edge[idx3x,idx3y,1])

    e2a = copy(env_ce.edge[idx2x,idx2y,2])
    e2b = copy(env_ce.edge[idx4x,idx4y,2])

    e3a = copy(env_ce.edge[idx1x,idx1y,3])
    e3b = copy(env_ce.edge[idx2x,idx2y,3])

    e4a = copy(env_ce.edge[idx3x,idx3y,4])
    e4b = copy(env_ce.edge[idx4x,idx4y,4])

    a1 = copy(ipeps_ntn.tensors[idx1x,idx1y])
    a2 = copy(ipeps_ntn.tensors[idx2x,idx2y])
    a3 = copy(ipeps_ntn.tensors[idx3x,idx3y])
    a4 = copy(ipeps_ntn.tensors[idx4x,idx4y])

    dim1 = chi
    dim2 = size(a1)[2]
    dim3 = size(a3)[2]

    x_tensor = zeros(Float64,dim2,dim1,dim3,dim1)
    y_tensor = zeros(Float64,dim2,dim1,dim3,dim1)

    uptensor = ein"myni,npu,pq,mqo -> yuio"(a1,e3a,c1,e1a)  
    downtensor = ein"myin,ntu,pt,mop -> yuio"(a3,e4a,c3,e1b) 
    x_tensor = ein"yumn,iomn -> yuio"(uptensor,downtensor)
    
    uptensor = ein"ymni,nup,mqo,pq- > yuio"(a2,e3b,e2a,c2)
    downtensor = ein"ymin,mop,nuq,pq -> yuio"(a4,e2b,e4b,c4)
    y_tensor = ein"yumn,iomn -> yuio"(uptensor,downtensor)

    x_matrix = reshape(x_tensor,(dim2*dim1,dim3*dim1))
    y_matrix = reshape(y_tensor,(dim2*dim1,dim3*dim1))

    x_matrix = x_matrix/maximum(abs.(x_matrix))
    y_matrix = y_matrix/maximum(abs.(y_matrix))

    x_matrix = x_matrix.*100
    y_matrix = y_matrix.*100

    x_matrix = x_matrix'
    y_matrix = y_matrix'
    
    R_matrix = y_matrix
    L_matrix = x_matrix
    R_matrix = R_matrix'


    u,s,v = MySVD.mysvd1(L_matrix*R_matrix)
    

    s1 = s[1:chi]
    u1 = u'[1:chi,:]
    v1 = v[:,1:chi] 
    s1_inv = sqrt.(1.0./s1)
    
    pt_matrix = R_matrix * v1 * Diagonal(s1_inv)
    qt_matrix = Diagonal(s1_inv) * u1 * L_matrix
    
    qt_matrix = qt_matrix'

    pt_tensor = reshape(pt_matrix,(dim2,chi,chi))
    qt_tensor = reshape(qt_matrix,(dim2,chi,chi))

   return qt_tensor,pt_tensor
end 

function GenIso_left(_index_,ipeps_ntn,env_ce)
  #  println("gradient test Iso step 1* ")

    chi = env_ce.chi
  
    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y

  #   println("gradient test Iso step 1 ")

   # idx1,idx2,idx3,idx4 = Genindex_ctmrg(m,n,_index_,"left")
   # idx1x,idx1y,idx2x,idx2y,idx3x,idx3y,idx4x,idx4y = Genxy(idx1,idx2,idx3,idx4,n)
    
    idx1x,idx1y = _index_[1],_index_[2] 
    idx2x,idx2y = idx1x,idx1y%4+1
    idx3x,idx3y = idx1x%4+1,idx1y
    idx4x,idx4y = idx1x%4+1,idx1y%4+1


    idx1x,idx1y = map_neel_pattern[idx1x,idx1y]
    idx2x,idx2y = map_neel_pattern[idx2x,idx2y]
    idx3x,idx3y = map_neel_pattern[idx3x,idx3y]
    idx4x,idx4y = map_neel_pattern[idx4x,idx4y]
 
  #  println("gradient test Iso step 2 ")
    c1 = copy(env_ce.corner[idx1x,idx1y,1])
    c2 = copy(env_ce.corner[idx2x,idx2y,2])
    c3 = copy(env_ce.corner[idx3x,idx3y,3])
    c4 = copy(env_ce.corner[idx4x,idx4y,4])

    e1a = copy(env_ce.edge[idx1x,idx1y,1])
    e1b = copy(env_ce.edge[idx3x,idx3y,1])

    e2a = copy(env_ce.edge[idx2x,idx2y,2])
    e2b = copy(env_ce.edge[idx4x,idx4y,2])

    e3a = copy(env_ce.edge[idx1x,idx1y,3])
    e3b = copy(env_ce.edge[idx2x,idx2y,3])

    e4a = copy(env_ce.edge[idx3x,idx3y,4])
    e4b = copy(env_ce.edge[idx4x,idx4y,4])

    a1 = copy(ipeps_ntn.tensors[idx1x,idx1y])
    a2 = copy(ipeps_ntn.tensors[idx2x,idx2y])
    a3 = copy(ipeps_ntn.tensors[idx3x,idx3y])
    a4 = copy(ipeps_ntn.tensors[idx4x,idx4y])

    dim1 = chi 
    dim2 = size(a1)[4]
    dim3 = size(a2)[4]

    x_tensor = zeros(Float64,dim3,dim1,dim2,dim1)
    y_tensor = zeros(Float64,dim3,dim1,dim2,dim1)

  #  println("gradient test Iso step 3 ")

    leftensor = ein"miny,mpu,qp,nqo -> yuio"(a1,e1a,c1,e3a)  
    # println(size(a2),size(e2a),size(c2),size(e3b))
    rightensor = ein"imny,mpu,qp,noq -> yuio"(a2,e2a,c2,e3b) 
    x_tensor = ein"iomn,yumn -> yuio"(leftensor,rightensor)  
    
    leftensor = ein"miyn,mup,pq,nqo -> yuio"(a3,e1b,c3,e4a)
    rightensor = ein"imyn,mup,pq,noq -> yuio"(a4,e2b,c4,e4b)
    y_tensor = ein"iomn,yumn -> yuio"(leftensor,rightensor)     

    # println("x ",size(x_tensor),"y ",size(y_tensor))

    x_matrix = reshape(x_tensor,(dim3*dim1,dim2*dim1))
    y_matrix = reshape(y_tensor,(dim3*dim1,dim2*dim1))

  #  println("gradient test Iso step 4 ")
     
    x_matrix = x_matrix/maximum(abs.(x_matrix))
    y_matrix = y_matrix/maximum(abs.(y_matrix))

    x_matrix = x_matrix.*100
    y_matrix = y_matrix.*100

    T_matrix = x_matrix
    D_matrix = y_matrix
    D_matrix = D_matrix' 

    # println("T ",size(T_matrix),"D ",size(D_matrix))


    u,s,v = MySVD.mysvd1(T_matrix*D_matrix)

    s1 = s[1:chi]
    u1 = u'[1:chi,:]
    v1 = v[:,1:chi] 
    s1_inv = sqrt.(1.0./s1)

    pl_matrix = D_matrix * v1 * Diagonal(s1_inv)
    ql_matrix = Diagonal(s1_inv) * u1 * T_matrix
    

    ql_matrix = ql_matrix'

    # println(size(pl_matrix))
    # println(size(a1),size(a2),size(a3),size(a4))
    pl_tensor = reshape(pl_matrix,(dim2,dim1,dim1))
    ql_tensor = reshape(ql_matrix,(dim2,dim1,dim1))
    
    return pl_tensor,ql_tensor
end




function GenIso_right(_index_,ipeps_ntn,env_ce::IPEPS_ENV)
    chi = env_ce.chi
  
    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y



    idx4x,idx4y = _index_[1],_index_[2]
    idx3x,idx3y = idx4x,idx4y-1 == 0 ? 4 : idx4y-1  
    idx1x,idx1y = idx3x-1 == 0 ? 4 : idx3x-1,idx3y   
    idx2x,idx2y = idx1x,idx1y%4+1
    

    idx1x,idx1y = map_neel_pattern[idx1x,idx1y]
    idx2x,idx2y = map_neel_pattern[idx2x,idx2y]
    idx3x,idx3y = map_neel_pattern[idx3x,idx3y]
    idx4x,idx4y = map_neel_pattern[idx4x,idx4y]


    c1 = copy(env_ce.corner[idx1x,idx1y,1])
    c2 = copy(env_ce.corner[idx2x,idx2y,2])
    c3 = copy(env_ce.corner[idx3x,idx3y,3])
    c4 = copy(env_ce.corner[idx4x,idx4y,4])

    e1a = copy(env_ce.edge[idx1x,idx1y,1])
    e1b = copy(env_ce.edge[idx3x,idx3y,1])

    e2a = copy(env_ce.edge[idx2x,idx2y,2])
    e2b = copy(env_ce.edge[idx4x,idx4y,2])

    e3a = copy(env_ce.edge[idx1x,idx1y,3])
    e3b = copy(env_ce.edge[idx2x,idx2y,3])

    e4a = copy(env_ce.edge[idx3x,idx3y,4])
    e4b = copy(env_ce.edge[idx4x,idx4y,4])

    a1 = copy(ipeps_ntn.tensors[idx1x,idx1y])
    a2 = copy(ipeps_ntn.tensors[idx2x,idx2y])
    a3 = copy(ipeps_ntn.tensors[idx3x,idx3y])
    a4 = copy(ipeps_ntn.tensors[idx4x,idx4y])

    dim1 = chi 
    dim2 = size(a1)[4]
    dim3 = size(a2)[4]
    
    x_tensor = zeros(Float64,dim3,dim1,dim2,dim1)
    y_tensor = zeros(Float64,dim3,dim1,dim2,dim1)

    leftensor = ein"miny,mpu,qp,nqo -> yuio"(a1,e1a,c1,e3a)  
    rightensor = ein"imny,mpu,qp,noq -> yuio"(a2,e2a,c2,e3b) 
    x_tensor =ein"iomn,yumn -> yuio"(leftensor,rightensor) 
    
    leftensor = ein"miyn,mup,pq,nqo -> yuio"(a3,e1b,c3,e4a) 
    rightensor = ein"imyn,mup,pq,noq -> yuio"(a4,e2b,c4,e4b) 
    y_tensor = ein"iomn,yumn -> yuio"(leftensor,rightensor)  

    x_matrix = reshape(x_tensor,(dim3*dim1,dim2*dim1))
    y_matrix = reshape(y_tensor,(dim3*dim1,dim2*dim1))
 
    
    # x_matrix = x_matrix.*100
    # y_matrix = y_matrix.*100
    x_matrix = x_matrix/maximum(abs.(x_matrix))
    y_matrix = y_matrix/maximum(abs.(y_matrix))

    x_matrix = x_matrix.*100
    y_matrix = y_matrix.*100

    x_matrix = x_matrix'
    y_matrix = y_matrix'
    
    L_matrix = x_matrix
    
    R_matrix = y_matrix 
    R_matrix = R_matrix'
    
    u,s,v = MySVD.mysvd1(L_matrix*R_matrix)

    s1 = s[1:chi]
    u1 = u'[1:chi,:]
    v1 = v[:,1:chi] 
    s1_inv = sqrt.(1.0./s1)
    
    p_matrix = R_matrix * v1 * Diagonal(s1_inv)
    q_matrix = Diagonal(s1_inv) * u1 * L_matrix
    
    q_matrix = q_matrix'
    
    p_tensor = reshape(p_matrix,(dim3,chi,chi))
    q_tensor = reshape(q_matrix,(dim3,chi,chi))
    return q_tensor,p_tensor
end

function down_move(tensors,env_ce::IPEPS_ENV,bond,id1,id2)
    tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 1, 2 ) 
    tensor_data = Zygote.Buffer(tensor_data1)
    tensor_data[1,1] = tensors[1,:,:,:,:,:]
    tensor_data[1,2] = tensors[2,:,:,:,:,:]
    # tensor_data[2,1] = tensors[3,:,:,:,:,:]
    # tensor_data[2,2] = tensors[4,:,:,:,:,:]
 
    unitcell= unit_cell(2,2)
    initial_ipeps = IPEPS(unitcell,2,bond,tensor_data) 
    ipeps_ntn = Gen_NTNIPEPS(initial_ipeps,id1,id2)




    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y
    # s = 0
    # bond = ipeps_ntn.bond_dim
    # chi = env_ce.chi

    p1 = Array{ Array{ Float64, 3 }, 1 }( undef, n )   
    q1 = Array{ Array{ Float64, 3 }, 1 }( undef, n )
    # println("n",n)
    p = Zygote.Buffer(p1)
    q = Zygote.Buffer(q1)
    p[:] = p1[:]
    q[:] = q1[:]



    for i = reverse(1:m)
       #GenIso s
       for j = 1:n 
           _index_ = (i,j) #n*(i-1) + j
           ptem,qtem = GenIso_down(_index_,ipeps_ntn,env_ce)
           p[j%n+1] = ptem
           q[j%n+1] = qtem
       end

       e4_new1 = Array{Any}(undef,n)
       e4_new = Zygote.Buffer(e4_new1)
       c3_new1 = Array{Any}(undef,n)
       c3_new = Zygote.Buffer(c3_new1)
       c4_new1 = Array{Any}(undef,n)
       c4_new = Zygote.Buffer(c4_new1)

       #renormalize
       for j = 1:n
        #    _index_ = n*(i-1) + j    
        #    index_update = _index_ - n <= 0 ? _index_ - n + m*n : _index_ - n # update index
        # #    Genxy(idx1,idx2,idx3,idx4,y)
           update_idx = i-1 ==0 ? m : i-1
           
           (i1,j1) = map_neel_pattern[i,j]
        #    println("第i行",i)
        #    println("第j个",j)
        #    println("p1q1 p2q2的号",j," ",j%n+1)
        #    println("更新标号",update_idx,j)

           p1 =  p[j]
           q1 =  q[j]
           p2 =  p[j%n + 1]
           q2 =  q[j%n + 1]

           a_tensor = ipeps_ntn.tensors[i1,j1] 
           c3 = env_ce.corner[i1,j1,3]
           e4 = env_ce.edge[i1,j1,4]
           c4 = env_ce.corner[i1,j1,4]
            
           e1 = env_ce.edge[i1,j1,1]
           e2 = env_ce.edge[i1,j1,2]
            
       #   c3_new = zeros(size(env_ce.corner[update_idx,j,3]))
       #   e4_new = zeros(size(env_ce.edge[update_idx,j,4]))
       #    c4_new = zeros(size(env_ce.corner[update_idx,j,4]))

           c3_newt = ein"mn,pym,pnu -> yu"(c3,e1,p1)   # @ein c3_new[1,2] := c3[m,n]*e1[p,1,m]*p1[p,n,2]
           e4_newt = ein"mku,mnyp,pkq,nqi -> yui"(q1,a_tensor,e4,p2)   # @ein e4_new[1,2,3] := q1[m,k,2]*a_tensor[m,n,1,p]*e4[p,k,q]*p2[n,q,3]
           c4_newt = ein"mnu,myp,pn -> yu"(q2,e2,c4)   #@ein c4_new[1,2] := q2[m,n,2]*e2[m,1,p]*c4[p,n]
           
           c3_newt = c3_newt/(norm(c3_newt)*0.01)
           c4_newt = c4_newt/(norm(c4_newt)*0.01)
           e4_newt = e4_newt/(norm(e4_newt)*0.01)

           c3_new[j] = deepcopy(c3_newt) 
           c4_new[j] = deepcopy(c4_newt)
           e4_new[j] = deepcopy(e4_newt)

        #    println(typeof(env_ce.edge))
        #    env_ce.corner[update_idx,j,3] = copy(c3_new1)
        #    env_ce.corner[update_idx,j,4] = copy(c4_new1)
        #    env_ce.edge[update_idx,j,4] = copy(e4_new1)
       end 
       for j = 1:n
           update_idx = i-1 ==0 ? m : i-1 
           idx_upd,j2 = map_neel_pattern[update_idx,j]

           env_ce.corner[idx_upd,j2,3] = deepcopy(c3_new[j])
           env_ce.corner[idx_upd,j2,4] = deepcopy(c4_new[j])
           env_ce.edge[idx_upd,j2,4] = deepcopy(e4_new[j])
       end


    end
   return env_ce#,s
end

function up_move(tensors,env_ce::IPEPS_ENV,bond,id1,id2)
    tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 1, 2 ) 
    tensor_data = Zygote.Buffer(tensor_data1)
    tensor_data[1,1] = tensors[1,:,:,:,:,:]
    tensor_data[1,2] = tensors[2,:,:,:,:,:]
    # tensor_data[2,1] = tensors[3,:,:,:,:,:]
    # tensor_data[2,2] = tensors[4,:,:,:,:,:]
 
    unitcell= unit_cell(2,2)
    initial_ipeps = IPEPS(unitcell,2,bond,tensor_data) 
    ipeps_ntn = Gen_NTNIPEPS(initial_ipeps,id1,id2)


    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y
    
    p1 = Array{ Array{ Float64, 3 }, 1 }( undef, n )   
    q1 = Array{ Array{ Float64, 3 }, 1 }( undef, n )
    p = Zygote.Buffer(p1)
    q = Zygote.Buffer(q1)
    p[:] = p1[:]
    q[:] = q1[:]

    e3_new1 = Array{Any}(undef,n)
    e3_new = Zygote.Buffer(e3_new1)
    c2_new1 = Array{Any}(undef,n)
    c2_new = Zygote.Buffer(c2_new1)
    c1_new1 = Array{Any}(undef,n)
    c1_new = Zygote.Buffer(c1_new1)

    for i = 1:m 
        for j = reverse(1:n)
            _index_ = (i,j)#(i-1)*n + j
            index_pq = n + 1 - j 
            ptem,qtem = GenIso_up(_index_,ipeps_ntn,env_ce)
            p[index_pq%n+1] = copy(ptem) 
            q[index_pq%n+1] = copy(qtem)
        end
        for j = reverse(1:n)
            update_idx = i%m + 1
            index_pq = n + 1 - j 
            
            
            
            # c1_new = zeros(size(env_ce.corner[update_idx,j,1]))
            # c2_new = zeros(size(env_ce.corner[update_idx,j,2]))
            # e3_new = zeros(size(env_ce.edge[update_idx,j,3]))    
            (i1,j1) = map_neel_pattern[i,j]
            a_tensor = ipeps_ntn.tensors[i1,j1]
            p1 =  p[index_pq]
            q1 =  q[index_pq]
            p2 =  p[index_pq%n + 1]
            q2 =  q[index_pq%n + 1]
             
            c1 = env_ce.corner[i1,j1,1]
            c2 = env_ce.corner[i1,j1,2]
            e3 = env_ce.edge[i1,j1,3]
             
            e1 = env_ce.edge[i1,j1,1]
            e2 = env_ce.edge[i1,j1,2] 

            c2_newt = ein"mny,mpu,np -> yu"(p1,e2,c2) 
            c1_newt = ein"mny,np,mpu -> yu"(q2,c1,e1) 
            e3_newt = ein"mnu,mpqy,qnk,pki -> yui"(p2,a_tensor,e3,q1)
             
            c2_newt = c2_newt/(norm(c2_newt)*0.01)
            c1_newt = c1_newt/(norm(c1_newt)*0.01)
            e3_newt = e3_newt/(norm(e3_newt)*0.01)

            c2_new[j] = deepcopy(c2_newt)
            c1_new[j] = deepcopy(c1_newt)
            e3_new[j] = deepcopy(e3_newt)
            # env_ce.corner[update_idx,j,1] = copy(c1_new)
            # env_ce.corner[update_idx,j,2] = copy(c2_new)
            # env_ce.edge[update_idx,j,3] = copy(e3_new)
        end
        for j = reverse(1:n)
           update_idx = i%m + 1
           idx_upd,j2 = map_neel_pattern[update_idx,j]

           env_ce.corner[idx_upd,j2,1] = copy(c1_new[j])
           env_ce.corner[idx_upd,j2,2] = copy(c2_new[j])
           env_ce.edge[idx_upd,j2,3] = copy(e3_new[j])
        end
    end
    return env_ce
end




function left_move(tensors,env_ce::IPEPS_ENV,bond,id1,id2)
    tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 1, 2 ) 
    tensor_data = Zygote.Buffer(tensor_data1)
    tensor_data[1,1] = tensors[1,:,:,:,:,:]
    tensor_data[1,2] = tensors[2,:,:,:,:,:]
    # tensor_data[2,1] = tensors[3,:,:,:,:,:]
    # tensor_data[2,2] = tensors[4,:,:,:,:,:]
 
    unitcell= unit_cell(2,2)
    initial_ipeps = IPEPS(unitcell,2,bond,tensor_data) 
    ipeps_ntn = Gen_NTNIPEPS(initial_ipeps,id1,id2)

  #  println("gradient test step 1 ")

    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y
    
    p1 = Array{ Array{ Float64, 3 }, 1 }( undef, m )   
    q1 = Array{ Array{ Float64, 3 }, 1 }( undef, m )
    p = Zygote.Buffer(p1)
    q = Zygote.Buffer(q1)
    p[:] = p1[:]
    q[:] = q1[:]


    e1_new1 = Array{Any}(undef,n)
    e1_new = Zygote.Buffer(e1_new1)
    c3_new1 = Array{Any}(undef,n)
    c3_new = Zygote.Buffer(c3_new1)
    c1_new1 = Array{Any}(undef,n)
    c1_new = Zygote.Buffer(c1_new1)



    for i = 1:n 
       for j = 1:m 
        #   println("gradient test step 2* ")
           _index_ = (j,i) #(j-1)*n+i 
        #   println("gradient test step 2** ")
        #   println(typeof(_index_))
           ptem,qtem = GenIso_left(_index_,ipeps_ntn,env_ce)
        #    println(j," ",i)
        #    println("chicun")
        #    println(size(ptem),size(qtem))
        #   println("gradient test step 2 ")

           p[j%m + 1] = copy(ptem)
           q[j%m + 1] = copy(qtem)
       end
       for j = 1:m 



        update_idx = i%n+1

     
        (j1,i1) = map_neel_pattern[j,i]
         
         
        p1 =  p[j]
        q1 =  q[j]
        p2 =  p[j%m + 1]
        q2 =  q[j%m + 1]  

        a_tensor = ipeps_ntn.tensors[j1,i1]

        c1 = env_ce.corner[j1,i1,1]
        e1 = env_ce.edge[j1,i1,1]
        c3 = env_ce.corner[j1,i1,3]
         
        e3 = env_ce.edge[j1,i1,3]
        e4 = env_ce.edge[j1,i1,4]

        c1_newt = ein"mn,pmy,pnu -> yu"(c1,e3,p1)  
        e1_newt = ein"mnu,pnk,pymq,qki -> yui"(q1,e1,a_tensor,p2) 
        c3_newt = ein"mny,np,mpu -> yu"(q2,c3,e4) 
         
        c1_newt = c1_newt/(norm(c1_newt)*0.01)
        e1_newt = e1_newt/(norm(e1_newt)*0.01)
        c3_newt = c3_newt/(norm(c3_newt)*0.01) 

        c1_new[j] = deepcopy(c1_newt)
        c3_new[j] = deepcopy(c3_newt)
        e1_new[j] = deepcopy(e1_newt)
     #   println("gradient test step 3 ") 

      #  update_idx = i%n+1
      #  j1,idx_upd = map_neel_pattern[j,update_idx] 
     #   env_ce.corner[j1,idx_upd,1] = deepcopy(c1_new[j])

     #   env_ce.corner[j1,idx_upd,3] = deepcopy(c3_new[j])
      #  env_ce.edge[j1,idx_upd,1] = deepcopy(e1_new[j])
       end
       for j = 1:m 
          update_idx = i%n+1
          j1,idx_upd = map_neel_pattern[j,update_idx] 
          env_ce.corner[j1,idx_upd,1] = deepcopy(c1_new[j])
          env_ce.corner[j1,idx_upd,3] = deepcopy(c3_new[j])
          env_ce.edge[j1,idx_upd,1] = deepcopy(e1_new[j])
       #   println("gradient test step 4 ")
       end
    end
    return env_ce
end

function right_move(tensors,env_ce::IPEPS_ENV,bond,id1,id2)
    tensor_data1 = Array{ Array{ Float64, 5 }, 2 }( undef, 1 , 2 ) 
    tensor_data = Zygote.Buffer(tensor_data1)
    tensor_data[1,1] = tensors[1,:,:,:,:,:]
    tensor_data[1,2] = tensors[2,:,:,:,:,:]
    # tensor_data[2,1] = tensors[3,:,:,:,:,:]
    # tensor_data[2,2] = tensors[4,:,:,:,:,:]
 
    unitcell= unit_cell(2,2)
    initial_ipeps = IPEPS(unitcell,2,bond,tensor_data) 
    ipeps_ntn = Gen_NTNIPEPS(initial_ipeps,id1,id2)


    m = ipeps_ntn.unitcell.x
    n = ipeps_ntn.unitcell.y
    
    p1 = Array{ Array{ Float64, 3 }, 1 }( undef, m )   
    q1 = Array{ Array{ Float64, 3 }, 1 }( undef, m )
    p = Zygote.Buffer(p1)
    q = Zygote.Buffer(q1)
    p[:] = p1[:]
    q[:] = q1[:]


    e2_new1 = Array{Any}(undef,n)
    e2_new = Zygote.Buffer(e2_new1)
    c4_new1 = Array{Any}(undef,n)
    c4_new = Zygote.Buffer(c4_new1)
    c2_new1 = Array{Any}(undef,n)
    c2_new = Zygote.Buffer(c2_new1)


    for i = reverse(1:n)#reverse
        for j = reverse(1:m)
            _index_ = (j,i)#(j-1)*n+i
            index_pq = m-j+1  
            ptem,qtem = GenIso_right(_index_,ipeps_ntn,env_ce)
  
            p[index_pq%n + 1] = copy(ptem)
            q[index_pq%n + 1] = copy(qtem)
        end
        for j = reverse(1:m)

            update_idx =  i-1 ==0 ? n : i-1
            index_pq = m-j+1   
             
            p1 =  p[index_pq]
            q1 =  q[index_pq]
            p2 =  p[index_pq%n + 1]
            q2 =  q[index_pq%n + 1]  


            j1,i1 = map_neel_pattern[j,i]


             
            a_tensor = ipeps_ntn.tensors[j1,i1]
             
            c2 = env_ce.corner[j1,i1,2]
            e2 = env_ce.edge[j1,i1,2]
            c4 = env_ce.corner[j1,i1,4]
             
            e3 = env_ce.edge[j1,i1,3] 
            e4 = env_ce.edge[j1,i1,4] 
             
            # c2_new = zeros(size(env_ce.corner[j,update_idx,2]))  
            # c4_new = zeros(size(env_ce.corner[j,update_idx,4]))  
            # e2_new = zeros(size(env_ce.edge[j,update_idx,2]))  

            c4_newt = ein"mny,mup,np -> yu"(p1,e4,c4)    
            c2_newt = ein"mnu,myp,pn -> yu"(q2,e3,c2)  
            e2_newt = ein"ymnp,pqi,nku,mkq -> yui"(a_tensor,q1,p2,e2)
           
            c4_newt = c4_newt/(norm(c4_newt)*0.01)
            c2_newt = c2_newt/(norm(c2_newt)*0.01)
            e2_newt = e2_newt/(norm(e2_newt)*0.01) 

            c4_new[j] = deepcopy(c4_newt)
            c2_new[j] = deepcopy(c2_newt)
            e2_new[j] = deepcopy(e2_newt)

            # env_ce.corner[j,update_idx,4] = copy(c4_new)
            # env_ce.corner[j,update_idx,2] = copy(c2_new)
            # env_ce.edge[j,update_idx,2] = copy(e2_new)
        end
        for j = reverse(1:m) 
            update_idx =  i-1 ==0 ? n : i-1

            j1,idx_upd = map_neel_pattern[j,update_idx]


            env_ce.corner[j1,idx_upd,4] = copy(c4_new[j])
            env_ce.corner[j1,idx_upd,2] = copy(c2_new[j])
            env_ce.edge[j1,idx_upd,2] = copy(e2_new[j])

        end
    end
    return env_ce
end





















