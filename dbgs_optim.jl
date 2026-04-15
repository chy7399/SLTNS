include("double_layer.jl")

function dbmain(;bond=3,chi = 20,iter_step = 5)
    ini_ipeps,ini_env,ham = pre_step(bond,chi,1)
    # f(x) = optim_gs_double(x,ini_env::IPEPS_ENV,ham,iter_step)



    start = zeros(4,2,bond,bond,bond,bond)
    start[1,:,:,:,:,:] = ini_ipeps.tensors[1,1]
    start[2,:,:,:,:,:] = ini_ipeps.tensors[1,2]
    start[3,:,:,:,:,:] = ini_ipeps.tensors[2,1]
    start[4,:,:,:,:,:] = ini_ipeps.tensors[2,2]
    
    corner = Zygote.Buffer(ini_env.corner)
    edge = Zygote.Buffer(ini_env.edge)
    corner[:,:,:] = ini_env.corner[:,:,:]
    edge[:,:,:] = ini_env.edge[:,:,:]

    unit = unit_cell(2,2)
    ini_env = IPEPS_ENV(unit,chi,corner,edge)
    f(x) = optim_gs_double(x,ini_env::IPEPS_ENV,ham,iter_step,bond)
    # println(typeof(conv_env.corner))

    # println(gradient(f, start)[1])
    result = optimize(f,x->gradient(f, x)[1],start,LBFGS(m=5)
    ,Optim.Options( show_trace = true,store_trace = true ,f_tol = 1e-6 ,iterations = 500 );inplace = false)
    return result
end

resut = dbmain()
# function testbk()
#     a = ini_IPEPS(phy=2,bond=4);
#     b = ini_bk_env(phy=2,bond=4,chi=50);
#     # c = Gen_NTNIPEPS(a,id1,id2);
#     # s = zeros(50)
#     c = GenWfbulk(a)
#  @time   for i = 1:40 
#         uc1,sc1,vc1 = svd(b.corner[2,2,1])
#         # stem = s
#         b = left_move(c,b)

#         println(b.corner[1,2,1][1,1])

#         b = up_move(c,b)

#         println(b.corner[1,2,1][1,1])

#         b = right_move(c,b)

#         println(b.corner[1,2,1][1,1])

#         b  = down_move(c,b)

#         println(b.corner[1,2,1][1,1])
         
#         uc,sc,vc = svd(b.corner[2,2,1])
#         println("finish",norm(sc-sc1))
#         # println("sconv",norm(s-stem))
#     end
#     ham = GetH( -1,0 )
#     @time  println(calculate_bk_energy(a,b,ham))
#     return nothing
# end

# testbk()
