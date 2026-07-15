from __future__ import annotations

from pathlib import Path

import pytest

from chemfluor.conforformer.dictionary import DictionaryError, load_conforformer_dictionary


def _write_dict(path: Path, tokens: list[str]) -> Path:
    path.write_text("\n".join(f"{token} 1" for token in tokens) + "\n", encoding="utf-8")
    return path


def test_token_order_indices_and_vocab_are_deterministic(tmp_path: Path) -> None:
    path = _write_dict(tmp_path / "dict.txt", ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "O"])
    first = load_conforformer_dictionary(path)
    second = load_conforformer_dictionary(path)
    assert first.tokens == ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "O")
    assert first.token_to_index == second.token_to_index == {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "C": 4, "O": 5}
    assert first.index_to_token[4] == "C"
    assert first.vocab_size == 6
    assert first.pad_id == 0
    assert first.cls_id == 1
    assert first.sep_id == 2
    assert first.unk_id == 3


def test_duplicate_tokens_fail(tmp_path: Path) -> None:
    path = _write_dict(tmp_path / "dict.txt", ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "C"])
    with pytest.raises(DictionaryError, match="duplicate token"):
        load_conforformer_dictionary(path)


@pytest.mark.parametrize("missing", ["[PAD]", "[CLS]", "[SEP]", "[UNK]"])
def test_missing_required_special_tokens_fail(tmp_path: Path, missing: str) -> None:
    tokens = [token for token in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C"] if token != missing]
    path = _write_dict(tmp_path / "dict.txt", tokens)
    with pytest.raises(DictionaryError, match=missing):
        load_conforformer_dictionary(path)


def test_content_hash_is_stable_and_path_independent(tmp_path: Path) -> None:
    tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C"]
    first_path = _write_dict(tmp_path / "a.txt", tokens)
    nested = tmp_path / "nested"
    nested.mkdir()
    second_path = _write_dict(nested / "b.txt", tokens)
    first = load_conforformer_dictionary(first_path)
    second = load_conforformer_dictionary(second_path)
    assert first.sha256 == second.sha256
    second_path.write_text(second_path.read_text(encoding="utf-8") + "O 1\n", encoding="utf-8")
    changed = load_conforformer_dictionary(second_path)
    assert changed.sha256 != first.sha256
