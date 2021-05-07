import torch
from IPython import embed

def foo(x, y):
    return 2 * x + y

# Run `foo` with the provided inputs and record the tensor operations
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
embed()