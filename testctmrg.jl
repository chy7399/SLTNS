include("cal_energy.jl")
include("gs_optim.jl")


function main()
    id1,id2 = Genconst(2,3)
    a = ini_IPEPS(phy=2,bond=3);
    b = ini_env(phy=2,bond=3,chi=20);
    c = Gen_NTNIPEPS(a,id1,id2);
    # s = zeros(50)
 @time   for i = 1:40 
        uc1,sc1,vc1 = svd(b.corner[2,2,1])
        # stem = s
        b = left_move(c,b)

        println(b.corner[1,2,1][1,1])

        b = up_move(c,b)

        println(b.corner[1,2,1][1,1])

        b = right_move(c,b)

        println(b.corner[1,2,1][1,1])

        b  = down_move(c,b)

        println(b.corner[1,2,1][1,1])
         
        uc,sc,vc = svd(b.corner[2,2,1])
        println("finish",norm(sc-sc1))
        # println("sconv",norm(s-stem))
    end
    # println(b.edge[4,2,1]-b.edge[4,1,1])
    # println(b.edge[4,1,1])
    ham = GetH( -1,0 )
    @time  println(calculate_energy(c,b,ham))
    return b
end

env = main();
