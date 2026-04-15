
function main(bond,chi,g,loadname,data_select)

    id1,id2 = Genconst(2,bond)
    initial_ipeps,initial_ntnipeps,conv_env,ham = pre_step(loadname,data_select,bond,chi,g,id1,id2,tol = 1e-10)



    # f(x) = optim_gs(x,conv_env,ham,15,bond,id1,id2)
    start = zeros(2,2,bond,bond,bond,bond)
    start[1,:,:,:,:,:] = initial_ipeps.tensors[1,1]
    start[2,:,:,:,:,:] = initial_ipeps.tensors[1,2]
    # start[3,:,:,:,:,:] = initial_ipeps.tensors[2,1]
    # start[4,:,:,:,:,:] = initial_ipeps.tensors[2,2]
    
    corner = Zygote.Buffer(conv_env.corner)
    edge = Zygote.Buffer(conv_env.edge)
    corner[:,:,:] = conv_env.corner[:,:,:]
    edge[:,:,:] = conv_env.edge[:,:,:]

    unit = unit_cell(4,4)
    conv_env = IPEPS_ENV(unit,chi,corner,edge)

    # println(typeof(conv_env.corner))
    println(conv_env.edge[1,1,1][1,1,1])
    f(x) = optim_gs1(x,conv_env,ham,3,bond,id1,id2)
   # println("f(x) test",f(start))
   # println("f(x) gradient test",gradient(f, start)[1])
   # stop!!!
    result = optimize(f,x->gradient(f, x)[1],start,LBFGS(m=5)
    ,Optim.Options( show_trace = true,store_trace = true ,f_tol = 1e-7 ,iterations = 20 );inplace = false)
    result = Optim.minimizer(result)
    # result = gradient(f, start)[1]
#    result = f(start)
#    println(result)
    return result
end
# result,env = main(7,98,1);

# # e = zeros(0)
# # x = zeros(0)
# # for chi = 10:10:70
# #     a,env = main(2,chi,1)
# #     append!(e,a)
# #     append!(x,chi)
# # end
# println("complished")


function savewf(name,wf)
   AA = Dict()
   AA["wf"] = wf 
   save(name,AA)
end





