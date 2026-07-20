from __future__ import annotations

from typing import Any

import numpy as np


class TargetMaskDataset:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def collater(self, items: list[dict[str, Any]]) -> Any:
        batched = self.dataset.collater(items)

        if not items or "target_mask" not in items[0]:
            return batched

        target_mask = np.stack(
            [
                np.asarray(item["target_mask"], dtype=np.bool_)
                for item in items
            ],
            axis=0,
        ).astype(np.bool_, copy=False)

        target = batched.get("target") if isinstance(batched, dict) else None
        try:
            import torch
        except ModuleNotFoundError:
            torch = None

        if torch is not None and torch.is_tensor(target):
            batched["target_mask"] = torch.as_tensor(
                target_mask,
                dtype=torch.bool,
                device=target.device,
            )
        else:
            batched["target_mask"] = target_mask

        return batched
