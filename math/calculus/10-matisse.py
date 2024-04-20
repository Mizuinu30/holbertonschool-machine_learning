#!/usr/bin/env python3
"""matisse module"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial."""
    if not isinstance(poly, list) or len(poly) ==0:
        return None

    if len(poly) ==1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    if not derivative:
        return [0]

    return derivative
