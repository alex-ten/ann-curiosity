using Flux
import Flux: params, softmax, glorot_uniform
import Random: seed!

nlabels = 5
nfeatures = 3
nunits = 3

x_fam = (
    [1, 0, 0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
)

begin
    seed!(1)

    label = Dense(nlabels => nunits, identity; init=glorot_uniform, bias=false)
    color = Dense(nfeatures => nunits, identity; init=glorot_uniform, bias=false)
    shape = Dense(nfeatures => nunits, identity; init=glorot_uniform, bias=false)
    orien = Dense(nfeatures => nunits, identity; init=glorot_uniform, bias=false)

    m = Parallel(softmax ∘ +; label, color, shape, orien)


end