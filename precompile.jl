using OMEinsum
using Optim
using LinearAlgebra
using Zygote
using JLD2 
using FileIO   
using DelimitedFiles
include("mysvd.jl")
using .MySVD
using Zygote:@adjoint 
using Zygote:@showgrad 

include("cal_energy.jl")
include("IPEPS.jl")
include("gs_optim.jl")
include("main.jl")
include("ctmrg.jl")


const Sz = [ 0.5 0; 0 -0.5 ];
const Sx = [ 0 0.5; 0.5 0 ];
const Sy = [ 0 -0.5; 0.5 0 ];

const map_neel_pattern = [(1,1) (1,2) (1,3) (1,4);(2,1) (2,2) (2,3) (2,4);(1,3) (1,4) (1,1) (1,2);(2,3) (2,4) (2,1) (2,2)];



