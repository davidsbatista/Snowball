from typing import Generator, TextIO


def blocks(files: TextIO, size: int = 65536) -> Generator[str, None, None]:
    """Read the file block-wise."""
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer
