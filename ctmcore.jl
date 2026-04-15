using ChainRulesCore

"""

using ChinRulesCore to solve CtmRG Gradient

"""
function ChainRulesCore.rrule(::typeof(optim_gs),x,conv_env,ham,iter_ctmrg_step,bond,id1,id2,grad_ctmrg_step)
    y = optim_gs(x,conv_env,ham,iter_ctmrg_step,bond,id1,id2)
    function optim_gs_pullback(y)
        f(x) = optim_gs(x,conv_env,ham,grad_ctmrg_step,bond,id1,id2)
        dx = 1*gradient(f,x)[1]
        return NoTangent(),dx,NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent()
    end 
    return y,optim_gs_pullback
end
