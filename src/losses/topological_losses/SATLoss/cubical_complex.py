"""Cubical complex computation module, adapted from the torch_topological library"""
import numpy as np
import torch
from torch import nn
from torch_topological.nn import PersistenceInformation
import gudhi

class CubicalComplex(nn.Module):
    def __init__(self, superlevel=False, dim=2):
        super().__init__()
        self.superlevel = superlevel
        self.dim = dim

    def forward(self, x):
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)
        else:
            dims = len(x.shape) - 2

        if dims == 0:
            return self._forward(x)
        elif dims == 1:
            return [self._forward(x_) for x_ in x]
        elif dims == 2:
            return [[self._forward(x__) for x__ in x_] for x_ in x]

    def _forward(self, x):
        if self.superlevel:
            x = -x

        # Detach and move to CPU for GUDHI (non-differentiable C++ backend)
        cubical_complex = gudhi.CubicalComplex(
            dimensions=list(x.shape),
            top_dimensional_cells=x.detach().cpu().flatten().numpy()
        ) 

        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        max_dim = len(x.shape)
        persistence_information = [
            self._extract_generators_and_diagrams(x, cofaces, dim) 
            for dim in range(0, max_dim)
        ]
        return persistence_information

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        device = x.device
        # ✅ Explicitly create on the same device as input x
        pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long, device=device
            )
            pairs = torch.cat((pairs, regular_pairs))
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long, device=device
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs, device=device).fill_(max_index)
            infinite_pairs = torch.stack((infinite_pairs, fake_destroyers), 1)
            pairs = torch.cat((pairs, infinite_pairs))

        return self._create_tensors_from_pairs(x, pairs, dim)

    def _create_tensors_from_pairs(self, x, pairs, dim):
        xs = x.shape
        device = x.device

        # ✅ Convert to CPU for NumPy, then back to x.device
        creators = torch.as_tensor(
            np.column_stack(np.unravel_index(pairs[:, 0].cpu().numpy(), xs)),
            dtype=torch.long, device=device
        )
        destroyers = torch.as_tensor(
            np.column_stack(np.unravel_index(pairs[:, 1].cpu().numpy(), xs)),
            dtype=torch.long, device=device
        )
        gens = torch.as_tensor(torch.hstack((creators, destroyers)), device=device)

        # Indexing x (on CUDA) with pairs (now on CUDA) keeps this on CUDA
        persistence_diagram = torch.stack((
            x.ravel()[pairs[:, 0]],
            x.ravel()[pairs[:, 1]]
        ), 1)

        return PersistenceInformation(
            pairing=gens,
            diagram=persistence_diagram,
            dimension=dim
        )