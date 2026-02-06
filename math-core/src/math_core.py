"""
SymmetryKit — Universal Math Core (Python Reference Implementation)
===================================================================
Platform-agnostic algebra: Vec3, Mat4, align_to_surface, platform_reflect,
MatrixStack.  Pure Python, no external dependencies.

Canonical formula: M_reflect = P @ S @ P^-1

Reference: core/math_foundations.md
"""

from __future__ import annotations
import math
import copy
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SK_PI            = math.pi
SK_DEG2RAD       = SK_PI / 180.0
SK_RAD2DEG       = 180.0 / SK_PI
SK_EPSILON       = 1e-6
SK_DOT_SINGULARITY = 0.99
SK_DET_EPSILON   = 1e-8


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------
class Vec3:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> Vec3:
        return self.__mul__(s)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_sq(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalized(self) -> Vec3:
        ln = self.length()
        if ln < SK_EPSILON:
            return Vec3(0, 0, 0)
        inv = 1.0 / ln
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return (abs(self.x - other.x) < SK_EPSILON
                and abs(self.y - other.y) < SK_EPSILON
                and abs(self.z - other.z) < SK_EPSILON)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ---------------------------------------------------------------------------
# Mat4 — 4x4 matrix, row-major: m[row][col]
#
# Layout:
#   Row 0: X basis   | Row 1: Y basis   | Row 2: Z basis   | Row 3: Translation
#   Points transform as: p' = p @ M  (row-vector, left-multiply)
# ---------------------------------------------------------------------------
class Mat4:
    __slots__ = ('m',)

    def __init__(self):
        self.m: List[List[float]] = [[0.0]*4 for _ in range(4)]

    @staticmethod
    def identity() -> Mat4:
        r = Mat4()
        r.m[0][0] = 1.0; r.m[1][1] = 1.0; r.m[2][2] = 1.0; r.m[3][3] = 1.0
        return r

    @staticmethod
    def translate(v: Vec3) -> Mat4:
        r = Mat4.identity()
        r.m[3][0] = v.x; r.m[3][1] = v.y; r.m[3][2] = v.z
        return r

    @staticmethod
    def scale(v: Vec3) -> Mat4:
        r = Mat4()
        r.m[0][0] = v.x; r.m[1][1] = v.y; r.m[2][2] = v.z; r.m[3][3] = 1.0
        return r

    @staticmethod
    def rotate_x(degrees: float) -> Mat4:
        rad = degrees * SK_DEG2RAD
        c, s = math.cos(rad), math.sin(rad)
        r = Mat4.identity()
        r.m[1][1] = c;  r.m[1][2] = s
        r.m[2][1] = -s; r.m[2][2] = c
        return r

    @staticmethod
    def rotate_y(degrees: float) -> Mat4:
        rad = degrees * SK_DEG2RAD
        c, s = math.cos(rad), math.sin(rad)
        r = Mat4.identity()
        r.m[0][0] = c;  r.m[0][2] = -s
        r.m[2][0] = s;  r.m[2][2] = c
        return r

    @staticmethod
    def rotate_z(degrees: float) -> Mat4:
        rad = degrees * SK_DEG2RAD
        c, s = math.cos(rad), math.sin(rad)
        r = Mat4.identity()
        r.m[0][0] = c;  r.m[0][1] = s
        r.m[1][0] = -s; r.m[1][1] = c
        return r

    @staticmethod
    def rotate(axis: str, degrees: float) -> Mat4:
        a = axis.upper()
        if a == 'X': return Mat4.rotate_x(degrees)
        if a == 'Y': return Mat4.rotate_y(degrees)
        if a == 'Z': return Mat4.rotate_z(degrees)
        return Mat4.identity()

    def __matmul__(self, other: Mat4) -> Mat4:
        """Matrix multiply: self @ other"""
        r = Mat4()
        for i in range(4):
            for j in range(4):
                r.m[i][j] = sum(self.m[i][k] * other.m[k][j] for k in range(4))
        return r

    def transform_point(self, p: Vec3) -> Vec3:
        """Transform point (w=1): p @ M"""
        w = p.x * self.m[0][3] + p.y * self.m[1][3] + p.z * self.m[2][3] + self.m[3][3]
        inv_w = 1.0 / w if abs(w) > SK_EPSILON else 1.0
        return Vec3(
            (p.x * self.m[0][0] + p.y * self.m[1][0] + p.z * self.m[2][0] + self.m[3][0]) * inv_w,
            (p.x * self.m[0][1] + p.y * self.m[1][1] + p.z * self.m[2][1] + self.m[3][1]) * inv_w,
            (p.x * self.m[0][2] + p.y * self.m[1][2] + p.z * self.m[2][2] + self.m[3][2]) * inv_w,
        )

    def transform_vector(self, v: Vec3) -> Vec3:
        """Transform direction (w=0): v @ upper3x3"""
        return Vec3(
            v.x * self.m[0][0] + v.y * self.m[1][0] + v.z * self.m[2][0],
            v.x * self.m[0][1] + v.y * self.m[1][1] + v.z * self.m[2][1],
            v.x * self.m[0][2] + v.y * self.m[1][2] + v.z * self.m[2][2],
        )

    def determinant(self) -> float:
        """Full 4x4 determinant via 2x2 sub-determinant method."""
        m = self.m
        s0 = m[0][0]*m[1][1] - m[1][0]*m[0][1]
        s1 = m[0][0]*m[1][2] - m[1][0]*m[0][2]
        s2 = m[0][0]*m[1][3] - m[1][0]*m[0][3]
        s3 = m[0][1]*m[1][2] - m[1][1]*m[0][2]
        s4 = m[0][1]*m[1][3] - m[1][1]*m[0][3]
        s5 = m[0][2]*m[1][3] - m[1][2]*m[0][3]

        c5 = m[2][2]*m[3][3] - m[3][2]*m[2][3]
        c4 = m[2][1]*m[3][3] - m[3][1]*m[2][3]
        c3 = m[2][1]*m[3][2] - m[3][1]*m[2][2]
        c2 = m[2][0]*m[3][3] - m[3][0]*m[2][3]
        c1 = m[2][0]*m[3][2] - m[3][0]*m[2][2]
        c0 = m[2][0]*m[3][1] - m[3][0]*m[2][1]

        return s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0

    def inverse(self) -> Mat4:
        """General 4x4 inverse via adjugate / determinant."""
        m = self.m

        s0 = m[0][0]*m[1][1] - m[1][0]*m[0][1]
        s1 = m[0][0]*m[1][2] - m[1][0]*m[0][2]
        s2 = m[0][0]*m[1][3] - m[1][0]*m[0][3]
        s3 = m[0][1]*m[1][2] - m[1][1]*m[0][2]
        s4 = m[0][1]*m[1][3] - m[1][1]*m[0][3]
        s5 = m[0][2]*m[1][3] - m[1][2]*m[0][3]

        c5 = m[2][2]*m[3][3] - m[3][2]*m[2][3]
        c4 = m[2][1]*m[3][3] - m[3][1]*m[2][3]
        c3 = m[2][1]*m[3][2] - m[3][1]*m[2][2]
        c2 = m[2][0]*m[3][3] - m[3][0]*m[2][3]
        c1 = m[2][0]*m[3][2] - m[3][0]*m[2][2]
        c0 = m[2][0]*m[3][1] - m[3][0]*m[2][1]

        det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0
        if abs(det) < SK_DET_EPSILON:
            return Mat4.identity()

        inv_det = 1.0 / det
        inv = Mat4()

        inv.m[0][0] = ( m[1][1]*c5 - m[1][2]*c4 + m[1][3]*c3) * inv_det
        inv.m[0][1] = (-m[0][1]*c5 + m[0][2]*c4 - m[0][3]*c3) * inv_det
        inv.m[0][2] = ( m[3][1]*s5 - m[3][2]*s4 + m[3][3]*s3) * inv_det
        inv.m[0][3] = (-m[2][1]*s5 + m[2][2]*s4 - m[2][3]*s3) * inv_det

        inv.m[1][0] = (-m[1][0]*c5 + m[1][2]*c2 - m[1][3]*c1) * inv_det
        inv.m[1][1] = ( m[0][0]*c5 - m[0][2]*c2 + m[0][3]*c1) * inv_det
        inv.m[1][2] = (-m[3][0]*s5 + m[3][2]*s2 - m[3][3]*s1) * inv_det
        inv.m[1][3] = ( m[2][0]*s5 - m[2][2]*s2 + m[2][3]*s1) * inv_det

        inv.m[2][0] = ( m[1][0]*c4 - m[1][1]*c2 + m[1][3]*c0) * inv_det
        inv.m[2][1] = (-m[0][0]*c4 + m[0][1]*c2 - m[0][3]*c0) * inv_det
        inv.m[2][2] = ( m[3][0]*s4 - m[3][1]*s2 + m[3][3]*s0) * inv_det
        inv.m[2][3] = (-m[2][0]*s4 + m[2][1]*s2 - m[2][3]*s0) * inv_det

        inv.m[3][0] = (-m[1][0]*c3 + m[1][1]*c1 - m[1][2]*c0) * inv_det
        inv.m[3][1] = ( m[0][0]*c3 - m[0][1]*c1 + m[0][2]*c0) * inv_det
        inv.m[3][2] = (-m[3][0]*s3 + m[3][1]*s1 - m[3][2]*s0) * inv_det
        inv.m[3][3] = ( m[2][0]*s3 - m[2][1]*s1 + m[2][2]*s0) * inv_det

        return inv

    def transposed(self) -> Mat4:
        r = Mat4()
        for i in range(4):
            for j in range(4):
                r.m[i][j] = self.m[j][i]
        return r

    def get_row(self, i: int) -> Vec3:
        return Vec3(self.m[i][0], self.m[i][1], self.m[i][2])

    def get_translation(self) -> Vec3:
        return self.get_row(3)

    def set_row(self, i: int, v: Vec3) -> None:
        self.m[i][0] = v.x; self.m[i][1] = v.y; self.m[i][2] = v.z

    def set_translation(self, v: Vec3) -> None:
        self.set_row(3, v)

    def copy(self) -> Mat4:
        r = Mat4()
        for i in range(4):
            for j in range(4):
                r.m[i][j] = self.m[i][j]
        return r

    def __repr__(self) -> str:
        rows = []
        for i in range(4):
            rows.append(f"  [{self.m[i][0]:8.4f} {self.m[i][1]:8.4f} {self.m[i][2]:8.4f} {self.m[i][3]:8.4f}]")
        return "Mat4(\n" + "\n".join(rows) + "\n)"


# ---------------------------------------------------------------------------
# AlignToSurface — Gram-Schmidt orthonormal basis
# Reference: core/math_foundations.md Section 1
# ---------------------------------------------------------------------------
def align_to_surface(hit_point: Vec3, normal: Vec3,
                     up: Vec3 = Vec3(0, 1, 0)) -> Mat4:
    """
    Build an orthonormal basis aligned to a surface normal.

    Y-axis = normal direction
    X-axis = orthogonal via Gram-Schmidt
    Z-axis = completes right-handed basis
    Origin = hit_point
    """
    y = normal.normalized()

    # Singularity guard
    temp = Vec3(1, 0, 0) if abs(y.dot(up)) > SK_DOT_SINGULARITY else up

    x = temp.cross(y).normalized()
    z = x.cross(y)  # Already unit length

    result = Mat4.identity()
    result.set_row(0, x)
    result.set_row(1, y)
    result.set_row(2, z)
    result.set_row(3, hit_point)
    return result


# ---------------------------------------------------------------------------
# PlatformReflect — Reflection across a platform's local axis
# Reference: core/math_foundations.md Section 2
#
# THE KEY FORMULA:  M_reflect = P @ S @ P^-1
# ---------------------------------------------------------------------------
_AXIS_SCALE = {
    'X':   Vec3(-1,  1,  1),
    'Y':   Vec3( 1, -1,  1),
    'Z':   Vec3( 1,  1, -1),
    'XY':  Vec3(-1, -1,  1),
    'XZ':  Vec3(-1,  1, -1),
    'YZ':  Vec3( 1, -1, -1),
    'XYZ': Vec3(-1, -1, -1),  # Point inversion
}


def platform_reflect(platform_matrix: Mat4, axis: str = 'X') -> Mat4:
    """
    Reflect across a platform's local axis.

    M_reflect = P @ S @ P^-1

    Supports: X, Y, Z, XY, XZ, YZ, XYZ
    """
    s = _AXIS_SCALE.get(axis.upper(), Vec3(-1, 1, 1))

    S     = Mat4.scale(s)
    P     = platform_matrix
    P_inv = P.inverse()

    return P @ S @ P_inv


# ---------------------------------------------------------------------------
# MatrixStack — Accumulated transformation with scope isolation
# Reference: core/state_schema.md
# ---------------------------------------------------------------------------
class MatrixStack:
    """
    Matrix stack providing push/pop scope isolation.

    Invariants:
      - Stack always has >= 1 element (base identity).
      - push() duplicates current; pop() restores parent.
      - transform(T) post-multiplies: current = current @ T
    """

    def __init__(self):
        self._stack: List[Mat4] = [Mat4.identity()]

    @property
    def current(self) -> Mat4:
        return self._stack[-1]

    @current.setter
    def current(self, value: Mat4) -> None:
        self._stack[-1] = value

    def push(self) -> None:
        self._stack.append(self.current.copy())

    def pop(self) -> None:
        if len(self._stack) > 1:
            self._stack.pop()

    def transform(self, T: Mat4) -> None:
        self._stack[-1] = self._stack[-1] @ T

    @property
    def depth(self) -> int:
        return len(self._stack)

    def reset(self) -> None:
        self._stack.clear()
        self._stack.append(Mat4.identity())


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Identity * Identity = Identity
    I = Mat4.identity()
    I2 = I @ I
    assert all(
        abs(I2.m[i][j] - (1.0 if i == j else 0.0)) < SK_EPSILON
        for i in range(4) for j in range(4)
    ), "Identity multiply failed"

    # Translation round-trip
    T = Mat4.translate(Vec3(10, 20, 30))
    origin = Vec3(0, 0, 0)
    moved = T.transform_point(origin)
    assert moved == Vec3(10, 20, 30), f"Translate failed: {moved}"

    # Inverse: T * T^-1 = I
    T_inv = T.inverse()
    should_be_I = T @ T_inv
    for i in range(4):
        for j in range(4):
            expected = 1.0 if i == j else 0.0
            assert abs(should_be_I.m[i][j] - expected) < 1e-5, \
                f"Inverse failed at [{i}][{j}]: {should_be_I.m[i][j]}"

    # PlatformReflect at identity = simple X flip
    M_ref = platform_reflect(Mat4.identity(), 'X')
    p = Vec3(5, 3, 2)
    p_reflected = M_ref.transform_point(p)
    assert p_reflected == Vec3(-5, 3, 2), f"X reflect failed: {p_reflected}"

    # AlignToSurface with Y-up normal gives identity rotation
    M_align = align_to_surface(Vec3(0, 0, 0), Vec3(0, 1, 0))
    # X-axis row should be (1,0,0) or (-1,0,0) depending on cross product
    # With normal=(0,1,0) and up=(0,1,0), singularity uses temp=(1,0,0)
    # temp.cross(y) = (1,0,0).cross(0,1,0) = (0,0,1) => x=(0,0,1)
    # z = x.cross(y) = (0,0,1).cross(0,1,0) = (-1,0,0)
    x_row = M_align.get_row(0)
    y_row = M_align.get_row(1)
    z_row = M_align.get_row(2)
    assert abs(x_row.dot(y_row)) < SK_EPSILON, "Align: X not orthogonal to Y"
    assert abs(x_row.dot(z_row)) < SK_EPSILON, "Align: X not orthogonal to Z"
    assert abs(y_row.dot(z_row)) < SK_EPSILON, "Align: Y not orthogonal to Z"

    # MatrixStack push/pop
    stack = MatrixStack()
    stack.transform(Mat4.translate(Vec3(10, 0, 0)))
    stack.push()
    stack.transform(Mat4.translate(Vec3(0, 5, 0)))
    inner_t = stack.current.get_translation()
    assert inner_t == Vec3(10, 5, 0), f"Stack inner: {inner_t}"
    stack.pop()
    outer_t = stack.current.get_translation()
    assert outer_t == Vec3(10, 0, 0), f"Stack outer: {outer_t}"

    print("All math_core.py self-tests passed.")
