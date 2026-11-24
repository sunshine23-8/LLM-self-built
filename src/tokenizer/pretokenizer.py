import re
from functools import lru_cache

# Splits each digit 0-9 individually while keeping word prefixes and math symbols clean.
PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def pretokenize(text: str) -> list[str]:
    return PRETOKENIZE_PATTERN.findall(text)
