"""
Decodes Valhalla's encoded route shapes.

Valhalla uses 6 digits of decimal precision (NOT the 5-digit Google Maps
default) - see https://valhalla.github.io/valhalla/decoding/. This is a
direct adaptation of the reference Python implementation given on that
page, so it matches Valhalla's own encoding exactly.
"""
from typing import List, Tuple

PRECISION_INV = 1.0 / 1e6
PRECISION = 1e6


def encode(coords: List[Tuple[float, float]]) -> str:
    """
    Encode a list of (lat, lon) tuples into a Valhalla-compatible polyline6
    string. This is the inverse of decode() below, adapted from the same
    reference algorithm Valhalla documents at
    https://valhalla.github.io/valhalla/decoding/ (6 digits of precision,
    NOT the 5-digit Google Maps default).
    """

    def _encode_value(value: int) -> str:
        value = value << 1
        if value < 0:
            value = ~value
        chunks = []
        while value >= 0x20:
            chunks.append((0x20 | (value & 0x1F)) + 63)
            value >>= 5
        chunks.append(value + 63)
        return "".join(chr(c) for c in chunks)

    result = []
    prev_lat, prev_lon = 0, 0
    for lat, lon in coords:
        lat_i = int(round(lat * PRECISION))
        lon_i = int(round(lon * PRECISION))
        result.append(_encode_value(lat_i - prev_lat))
        result.append(_encode_value(lon_i - prev_lon))
        prev_lat, prev_lon = lat_i, lon_i
    return "".join(result)


def decode(encoded: str) -> List[Tuple[float, float]]:
    """Decode a Valhalla polyline6 string into a list of (lat, lon) tuples."""
    decoded: List[Tuple[float, float]] = []
    previous = [0, 0]
    i = 0
    length = len(encoded)

    while i < length:
        ll = [0, 0]
        for j in range(2):
            shift = 0
            byte = 0x20
            while byte >= 0x20:
                byte = ord(encoded[i]) - 63
                i += 1
                ll[j] |= (byte & 0x1F) << shift
                shift += 5
            ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
            previous[j] = ll[j]
        decoded.append((round(ll[0] * PRECISION_INV, 6), round(ll[1] * PRECISION_INV, 6)))

    return decoded
