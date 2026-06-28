from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


Row = dict[str, Any]


@dataclass(frozen=True)
class ExpandingWindowSpec:
    min_train_dates: int
    test_window_dates: int
    embargo_dates: int = 0

    def validate(self) -> None:
        if self.min_train_dates < 1:
            raise ValueError("min_train_dates must be at least one")
        if self.test_window_dates < 1:
            raise ValueError("test_window_dates must be at least one")
        if self.embargo_dates < 0:
            raise ValueError("embargo_dates cannot be negative")


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    embargoed_dates: tuple[str, ...]
    train_rows: tuple[Row, ...]
    test_rows: tuple[Row, ...]

    @property
    def chronological_guard_passed(self) -> bool:
        return bool(
            self.train_dates
            and self.test_dates
            and self.train_dates[-1] < self.test_dates[0]
        )


class ExpandingWindowSplitter:
    def __init__(self, spec: ExpandingWindowSpec):
        spec.validate()
        self.spec = spec

    def split(
        self,
        rows: Sequence[Row],
        *,
        dates: Sequence[str] | None = None,
        date_getter: Callable[[Row], str] | None = None,
    ) -> list[WalkForwardFold]:
        get_date = date_getter or (lambda row: str(row["rebalance_date"]))
        ordered_dates = list(dates or sorted({get_date(row) for row in rows}))
        first_test_index = self.spec.min_train_dates + self.spec.embargo_dates
        folds = []
        for fold_id, test_start in enumerate(
            range(first_test_index, len(ordered_dates), self.spec.test_window_dates),
            start=1,
        ):
            train_dates = tuple(
                ordered_dates[: test_start - self.spec.embargo_dates]
            )
            test_dates = tuple(
                ordered_dates[test_start : test_start + self.spec.test_window_dates]
            )
            embargoed_dates = tuple(
                ordered_dates[test_start - self.spec.embargo_dates : test_start]
            )
            train_set = set(train_dates)
            test_set = set(test_dates)
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_dates=train_dates,
                    test_dates=test_dates,
                    embargoed_dates=embargoed_dates,
                    train_rows=tuple(row for row in rows if get_date(row) in train_set),
                    test_rows=tuple(row for row in rows if get_date(row) in test_set),
                )
            )
        return folds
