# Math

> Math utilities, geometry tools, and procedural frameworks for 3D applications

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Last Commit](https://img.shields.io/github/last-commit/JohnnyNeoman/Math)](https://github.com/JohnnyNeoman/Math)

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| [math-core](./math-core/) | Zero-dependency vector/matrix library | Active |
| [procedural-framework](./procedural-framework/) | Skeletal Singleton Tree system | Active |
| [ml-geometry](./ml-geometry/) | ML operators for geometry | WIP |
| [maya-math](./maya-math/) | Maya-specific utilities | Planned |

## Quick Start

```python
from math_core.src.math_core import Vec3, Mat4, MatrixStack

v = Vec3(1, 0, 0)
m = Mat4.rotate_y(45)
result = m.transform_point(v)
```

## Features

- **Pure Python** - No external dependencies for core math
- **Cross-Platform** - Works with Maya, Blender, UE5
- **Compositional** - Functional node algebra for procedural geometry
- **ML-Ready** - Neural operator integration specs

## Install

```bash
git clone https://github.com/JohnnyNeoman/Math.git
cd Math
pip install -e .
```

## Documentation

- [Math Foundations](./math-core/docs/math-foundations.md)
- [Quick Reference](./math-core/docs/quick-reference.md)
- [Node Algebra](./procedural-framework/docs/node-algebra.md)

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- GitHub: [@JohnnyNeoman](https://github.com/JohnnyNeoman)
