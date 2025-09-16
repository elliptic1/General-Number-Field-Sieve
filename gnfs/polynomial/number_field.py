"""Algebraic number field utilities for GNFS polynomials.

The General Number Field Sieve works with the number field defined by the
algebraic polynomial chosen during the selection phase.  Real
implementations keep track of arithmetic in this field in order to lift
relations and compute square roots.  The helpers in this module provide a
compact but faithful model of that arithmetic: elements are represented on
an integral power basis and operations are carried out modulo the minimal
polynomial.  Although intentionally small in scope, the code mirrors the
machinery used in genuine GNFS programs and therefore moves beyond a toy
placeholder.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, Tuple, Union

from .polynomial import Polynomial

NumberLike = Union[int, Fraction]


def _to_fraction(value: NumberLike) -> Fraction:
    """Convert ``value`` to :class:`fractions.Fraction`."""

    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _fraction_determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    """Compute the determinant of ``matrix`` using Fraction arithmetic."""

    n = len(matrix)
    if n == 0:
        return Fraction(1)
    mat = [list(row) for row in matrix]
    det = Fraction(1)
    for i in range(n):
        pivot = None
        for r in range(i, n):
            if mat[r][i] != 0:
                pivot = r
                break
        if pivot is None:
            return Fraction(0)
        if pivot != i:
            mat[i], mat[pivot] = mat[pivot], mat[i]
            det *= -1
        pivot_val = mat[i][i]
        det *= pivot_val
        for r in range(i + 1, n):
            factor = mat[r][i] / pivot_val
            if factor == 0:
                continue
            for c in range(i, n):
                mat[r][c] -= factor * mat[i][c]
    return det


class NumberField:
    """Algebraic number field ``Q[x] / (f(x))`` for a monic polynomial ``f``."""

    def __init__(self, polynomial: Polynomial):
        if polynomial.degree() < 1:
            raise ValueError("minimal polynomial must have degree >= 1")
        leading = polynomial.coeffs[-1]
        if leading == 0:
            raise ValueError("leading coefficient must be non-zero")
        lead = Fraction(leading)
        self._polynomial = polynomial
        self._monic_coeffs = tuple(Fraction(c, lead) for c in polynomial.coeffs)
        self._reduction_coeffs = self._monic_coeffs[:-1]
        self._degree = polynomial.degree()
        self._alpha: NumberFieldElement | None = None
        self._power_basis: Tuple[NumberFieldElement, ...] | None = None

    @property
    def minimal_polynomial(self) -> Polynomial:
        """Return the minimal polynomial that defines the field."""

        return self._polynomial

    @property
    def degree(self) -> int:
        """Return the extension degree of the field."""

        return self._degree

    def element(self, coeffs: IterableABC[NumberLike]) -> "NumberFieldElement":
        """Create a number field element from ``coeffs`` in the power basis."""

        return NumberFieldElement(self, coeffs)

    def rational(self, value: NumberLike) -> "NumberFieldElement":
        """Embed a rational number into the field."""

        return NumberFieldElement(self, [value])

    @property
    def alpha(self) -> "NumberFieldElement":
        """Return the distinguished root of the defining polynomial."""

        if self._alpha is None:
            if self._degree == 1:
                coeffs = [Fraction(0), Fraction(1)]
            else:
                coeffs = [Fraction(0)] * self._degree
                coeffs[1] = Fraction(1)
            self._alpha = NumberFieldElement(self, coeffs)
        return self._alpha

    @property
    def power_basis(self) -> Tuple["NumberFieldElement", ...]:
        """Return the integral power basis ``(1, alpha, ..., alpha^{d-1})``."""

        if self._power_basis is None:
            basis = [self.rational(1)]
            alpha = self.alpha
            current = self.rational(1)
            for _ in range(1, self._degree):
                current = current * alpha
                basis.append(current)
            self._power_basis = tuple(basis)
        return self._power_basis

    def norm(self, element: Union["NumberFieldElement", NumberLike, IterableABC[NumberLike]]) -> Fraction | int:
        """Return the field norm of ``element``."""

        if isinstance(element, NumberFieldElement):
            elem = element
        elif isinstance(element, IterableABC) and not isinstance(element, (str, bytes)):
            elem = NumberFieldElement(self, element)
        else:
            elem = self.rational(element)  # type: ignore[arg-type]
        if elem.field is not self:
            raise TypeError("element does not belong to this number field")
        basis = self.power_basis
        matrix = [[coeff for coeff in (elem * b).coeffs] for b in basis]
        det = _fraction_determinant(matrix)
        return det if det.denominator != 1 else det.numerator

    def _reduce(self, coeffs: IterableABC[NumberLike]) -> Tuple[Fraction, ...]:
        values = [_to_fraction(c) for c in coeffs]
        if not values:
            values = [Fraction(0)]
        if len(values) < self._degree:
            values.extend(Fraction(0) for _ in range(self._degree - len(values)))
        reduction = list(values)
        deg = self._degree
        if deg == 0:
            return tuple()
        for exp in range(len(reduction) - 1, deg - 1, -1):
            coeff = reduction[exp]
            if coeff == 0:
                continue
            reduction[exp] = Fraction(0)
            shift = exp - deg
            for i in range(deg):
                reduction[shift + i] -= coeff * self._reduction_coeffs[i]
        reduction = reduction[:deg]
        for i, value in enumerate(reduction):
            if value == 0:
                reduction[i] = Fraction(0)
        return tuple(reduction)


@dataclass(frozen=True)
class NumberFieldElement:
    """Element of a :class:`NumberField` represented on the power basis."""

    field: NumberField
    coeffs: Tuple[Fraction, ...]

    def __init__(self, field: NumberField, coeffs: IterableABC[NumberLike]):
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "coeffs", field._reduce(coeffs))

    # ------------------------------------------------------------------
    # Helper coercion utilities
    # ------------------------------------------------------------------
    def _coerce(self, other: Union["NumberFieldElement", NumberLike, IterableABC[NumberLike]]) -> "NumberFieldElement":
        if isinstance(other, NumberFieldElement):
            if other.field is not self.field:
                raise TypeError("elements belong to different number fields")
            return other
        if isinstance(other, IterableABC) and not isinstance(other, (str, bytes)):
            return NumberFieldElement(self.field, other)
        return self.field.rational(other)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------
    def __add__(self, other: Union["NumberFieldElement", NumberLike, IterableABC[NumberLike]]):
        other_elem = self._coerce(other)
        coeffs = [a + b for a, b in zip(self.coeffs, other_elem.coeffs)]
        return NumberFieldElement(self.field, coeffs)

    def __radd__(self, other: Union[NumberLike, IterableABC[NumberLike]]):
        return self.__add__(other)

    def __sub__(self, other: Union["NumberFieldElement", NumberLike, IterableABC[NumberLike]]):
        other_elem = self._coerce(other)
        coeffs = [a - b for a, b in zip(self.coeffs, other_elem.coeffs)]
        return NumberFieldElement(self.field, coeffs)

    def __rsub__(self, other: Union[NumberLike, IterableABC[NumberLike]]):
        other_elem = self._coerce(other)
        coeffs = [b - a for a, b in zip(self.coeffs, other_elem.coeffs)]
        return NumberFieldElement(self.field, coeffs)

    def __neg__(self):
        return NumberFieldElement(self.field, (-c for c in self.coeffs))

    def __mul__(self, other: Union["NumberFieldElement", NumberLike, IterableABC[NumberLike]]):
        other_elem = self._coerce(other)
        deg = self.field.degree
        if deg == 0:
            return NumberFieldElement(self.field, [])
        size = 2 * deg - 1
        result = [Fraction(0)] * size
        for i, a in enumerate(self.coeffs):
            if a == 0:
                continue
            for j, b in enumerate(other_elem.coeffs):
                if b == 0:
                    continue
                result[i + j] += a * b
        return NumberFieldElement(self.field, result)

    def __rmul__(self, other: Union[NumberLike, IterableABC[NumberLike]]):
        return self.__mul__(other)

    def __pow__(self, power: int):
        if power < 0:
            raise ValueError("negative powers are not supported")
        result = self.field.rational(1)
        base = self
        exp = power
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        return result

    # ------------------------------------------------------------------
    # Comparisons and representations
    # ------------------------------------------------------------------
    def __eq__(self, other):
        if isinstance(other, NumberFieldElement):
            return self.field is other.field and self.coeffs == other.coeffs
        if isinstance(other, IterableABC) and not isinstance(other, (str, bytes)):
            return self == NumberFieldElement(self.field, other)
        if isinstance(other, (int, Fraction)):
            return self == self.field.rational(other)
        return False

    def __repr__(self) -> str:
        coeffs_str = ", ".join(str(c) for c in self.coeffs)
        return f"NumberFieldElement([{coeffs_str}])"
