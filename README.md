# Torch Geopooling - The geospatial pooling library for PyTorch

The Torch Geopooling library is an extension for PyTorch library that provide extra layers for
building geospatial neural networks.

## Installation

The library is distributed as PyPI package, to install that package, execute the following
command:
```sh
pip install torch_geopooling
```

You can use the `torch_geopooling` library for building neural networks with geospatial indexing.
The interface of the provided modules is compatible with [PyTorch](https://pytorch.org) library,
including automatic gradient computation.

## Documentation

The [Torch Geopooling Documentation](https://torch-geopooling.readthedocs.org) contains additional
details on how to get started with this library as well a few examples of training neural networks
that use geo-pooling modules.

## Usage

The module provides adaptive and regular modules that implement decomposition of point coordinates
in 2-dimensional space. Decomposition in this context implies separation of the space into
rectangles (quads).

Adaptive modules are building the decomposition during the training, while for regular modules
the decomposition should be computed beforehand. As a result, adaptive module builds sparse
decomposition, while regular module builds dense (regular) decomposition.

Using adaptive decomposition module for [EPSG:4326](https://epsg.io/4326) coordinates:
```py
import torch
from torch_geopooling.nn import AdaptiveQuadPool2d

# Create 5-feature vector for each node in a decomposition.
pool = AdaptiveQuadPool2d(5, (-180, -90, 360, 180), max_depth=12, capacity=10)
input = torch.DoubleTensor(1024, 2).uniform_(-90, 90)
output = pool(input)
```

Using regular decomposition module for arbitrary polygon:
```py
import torch
from shapely import Polygon
from torch_geopooling.nn import QuadPool2d

# Polygon for regular decomposition should be within an exterior boundary.
poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
exterior = (-100.0, -100.0, 200.0, 200.0)
# Create 3-feature vector for each node in a decomposition.
pool = QuadPool2d(3, poly, exterior, max_depth=10)
input = torch.DoubleTensor(200, 2).uniform_(0.0, 10.0)
output = pool(input)
```

## License

The Torch Geopooling is distributed under GPLv3 license. See the [LICENSE](LICENSE) file for full
license text.
