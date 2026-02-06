# math-core

> Zero-dependency vector and matrix library for 3D applications (Python)

## Features

- `Vec3` - 3D vector with dot, cross, normalize
- `Mat4` - 4x4 affine matrix with full transform operations
- `MatrixStack` - Push/pop scope isolation
- `align_to_surface()` - Gram-Schmidt orthonormalization
- `platform_reflect()` - Mirror formula M = P x S x P^-1

## Usage

```python
from math_core.src.math_core import Vec3, Mat4

v = Vec3(1, 2, 3)
m = Mat4.translate(10, 0, 0) @ Mat4.rotate_y(45)
result = m.transform_point(v)
```

## Documentation

- [Math Foundations](./docs/math-foundations.md) - Core mathematical concepts
- [State Schema](./docs/state-schema.md) - Matrix stack semantics
- [Mutation Schema](./docs/mutation-schema.md) - Geometry operations
- [Quick Reference](./docs/quick-reference.md) - Cheat sheet
