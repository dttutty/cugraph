#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, Lei Zhao
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_rank_log(path: Path, expected_rank: int) -> dict[str, Any]:
    setup = None
    done = False
    samples: dict[int, dict[str, Any]] = {}

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if row.get("rank") != expected_rank:
                raise ValueError(
                    f"{path}:{line_number}: expected rank {expected_rank}, got {row.get('rank')}"
                )
            event = row.get("event")
            if event == "setup":
                if setup is not None:
                    raise ValueError(f"{path}: duplicate setup event")
                setup = row
            elif event == "sample" and not row.get("warmup", False):
                iteration = row.get("iter")
                if not isinstance(iteration, int):
                    raise ValueError(f"{path}:{line_number}: sample has no integer iter")
                if iteration in samples:
                    raise ValueError(f"{path}: duplicate measured iteration {iteration}")
                samples[iteration] = row
            elif event == "done":
                done = True

    if setup is None:
        raise ValueError(f"{path}: missing setup event")
    if not samples:
        raise ValueError(f"{path}: no measured sample events")
    if not done:
        raise ValueError(f"{path}: missing done event")
    return {"setup": setup, "samples": samples}


def find_rank_log(log_dir: Path, rank: int) -> Path:
    names = [f"rank{rank}.log"]
    if rank == 0:
        names.append("markov_rank0.log")
    elif rank == 1:
        names.append("sun_rank1.log")
    matches = [log_dir / name for name in names if (log_dir / name).is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"{log_dir}: expected exactly one rank-{rank} log, found {len(matches)}"
        )
    return matches[0]


def summarize(log_dir: Path) -> dict[str, Any]:
    ranks = {
        rank: load_rank_log(find_rank_log(log_dir, rank), rank) for rank in (0, 1)
    }
    iteration_sets = [set(ranks[rank]["samples"]) for rank in (0, 1)]
    if iteration_sets[0] != iteration_sets[1]:
        raise ValueError(
            f"{log_dir}: rank iteration sets differ: {iteration_sets[0]} != {iteration_sets[1]}"
        )

    rank_max_seconds = []
    total_edges = []
    for iteration in sorted(iteration_sets[0]):
        rows = [ranks[rank]["samples"][iteration] for rank in (0, 1)]
        try:
            rank_max_seconds.append(max(float(row["seconds"]) for row in rows))
            total_edges.append(sum(int(row["minors"]) for row in rows))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{log_dir}: invalid sample row for iteration {iteration}") from error

    if len(set(total_edges)) != 1:
        raise ValueError(f"{log_dir}: sampled edge count changes across iterations")

    return {
        "logdir": str(log_dir),
        "iters": len(rank_max_seconds),
        "mean_rank_max_s": statistics.mean(rank_max_seconds),
        "median_rank_max_s": statistics.median(rank_max_seconds),
        "min_rank_max_s": min(rank_max_seconds),
        "max_rank_max_s": max(rank_max_seconds),
        "sample_edges_per_iter": total_edges[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize two-rank MG sampling benchmark logs."
    )
    parser.add_argument("logdirs", nargs="+", type=Path)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare exactly two directories as baseline then candidate.",
    )
    args = parser.parse_args()
    if args.compare and len(args.logdirs) != 2:
        parser.error("--compare requires exactly two log directories")

    summaries = [summarize(path) for path in args.logdirs]
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))

    if args.compare:
        baseline, candidate = summaries
        baseline_seconds = baseline["mean_rank_max_s"]
        candidate_seconds = candidate["mean_rank_max_s"]
        if baseline["sample_edges_per_iter"] != candidate["sample_edges_per_iter"]:
            raise ValueError("baseline and candidate sampled edge counts differ")
        comparison = {
            "baseline": baseline["logdir"],
            "candidate": candidate["logdir"],
            "speedup": baseline_seconds / candidate_seconds,
            "time_change_percent": 100.0
            * (candidate_seconds - baseline_seconds)
            / baseline_seconds,
        }
        print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
