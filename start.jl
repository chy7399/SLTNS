include("precompile.jl")
#include("ctmcore.jl")

function start(optimstep,d,chi)
    for i = 1:1:optimstep
        data_select = "default"

        if i == 1
            data_select = "rand_data"
        end
        println(data_select)
        loadname = "datafile/op$(i-1)wf_d$(d)chi$(chi).jld2"
        wf = main(d,chi,1,loadname,data_select)
        savename = "datafile/op$(i)wf_d$(d)chi$(chi).jld2"
        savewf(savename,wf)
    end
end

start(40,4,32)


