#!/usr/bin/env python3
"""poly_integral: Calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial"""


    # Validate input types
    if not isinstance(poly, list) or not poly:
        return None

    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    if not isinstance(C, (int, float)):
        return None

    # Handle the zero polynomial special case
    if poly == [0]:
        return [C]

    # Calculate the integral
    integral = [C]
    for power, coeff in enumerate(poly):
        new_coeff = coeff / (power + 1)
        integral.append(int(new_coeff) if new_coeff.is_integer() else new_coeff)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
