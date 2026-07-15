"""Lightweight ConforFormer dictionary loading without Uni-Core."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


REQUIRED_SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[UNK]")


class DictionaryError(ValueError):
    """Raised when a ConforFormer dictionary file is invalid."""


@dataclass(frozen=True)
class ConforFormerDictionary:
    """Ordered token dictionary matching Uni-Mol text dictionary semantics."""

    path: Path
    tokens: tuple[str, ...]
    token_to_index: dict[str, int]
    sha256: str

    def __post_init__(self) -> None:
        for token in REQUIRED_SPECIAL_TOKENS:
            if token not in self.token_to_index:
                raise DictionaryError(f"required special token missing from dictionary: {token}")

    @property
    def index_to_token(self) -> dict[int, str]:
        return {index: token for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def index(self, token: str) -> int:
        return self.token_to_index[token]

    @property
    def pad_id(self) -> int:
        return self.index("[PAD]")

    @property
    def cls_id(self) -> int:
        return self.index("[CLS]")

    @property
    def sep_id(self) -> int:
        return self.index("[SEP]")

    @property
    def unk_id(self) -> int:
        return self.index("[UNK]")


def load_conforformer_dictionary(path: Path | str) -> ConforFormerDictionary:
    """Load an ordered ConforFormer/Uni-Mol dictionary text file."""

    dictionary_path = Path(path)
    content = dictionary_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    tokens: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        token = stripped.split()[0]
        if token in seen:
            raise DictionaryError(f"duplicate token in dictionary at line {line_number}: {token}")
        seen.add(token)
        tokens.append(token)

    token_to_index = {token: index for index, token in enumerate(tokens)}
    return ConforFormerDictionary(
        path=dictionary_path,
        tokens=tuple(tokens),
        token_to_index=token_to_index,
        sha256=digest,
    )
