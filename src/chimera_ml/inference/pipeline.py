from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from chimera_ml.inference.context import InferenceContext
from chimera_ml.inference.steps.base import BaseInferenceStep


def _prepare_nodes(nodes: list["InferenceGraphNode"]) -> dict[str, "InferenceGraphNode"]:
    # validates and indexes nodes
    known_ids = {node.node_id for node in nodes}
    if len(known_ids) != len(nodes):
        counts: dict[str, int] = {}
        for node in nodes:
            counts[node.node_id] = counts.get(node.node_id, 0) + 1

        duplicates = [step_id for step_id, count in counts.items() if count > 1]
        raise ValueError(
            "Inference step ids must be unique. "
            f"Duplicate node ids: {sorted(set(duplicates))}. "
            "If you reuse the same step name multiple times, set explicit 'id' values."
        )

    for node in nodes:
        if node.node_id in node.after:
            raise ValueError(f"Inference step '{node.node_id}' cannot depend on itself.")

        for dependency_id in node.after:
            if dependency_id not in known_ids:
                raise ValueError(f"Inference step '{node.node_id}' depends on unknown step '{dependency_id}'.")

    status: dict[str, int] = {node.node_id: 0 for node in nodes}
    node_map = {node.node_id: node for node in nodes}

    def _visit(node_id: str) -> None:
        if status[node_id] == 1:
            raise ValueError(f"Inference pipeline dependency cycle detected at step '{node_id}'.")
        
        if status[node_id] == 2:
            return

        status[node_id] = 1
        for dependency_id in node_map[node_id].after:
            _visit(dependency_id)
        
        status[node_id] = 2

    for node in nodes:
        _visit(node.node_id)

    return node_map


@dataclass(frozen=True)
class InferenceGraphNode:
    """Single inference step plus its DAG dependencies."""

    node_id: str
    step: BaseInferenceStep
    after: tuple[str, ...] = ()


class InferencePipeline:
    """Inference pipeline represented as a small DAG of step dependencies."""

    def __init__(
        self,
        nodes: list[InferenceGraphNode],
        *,
        name: str = "inference_pipeline",
        parallel_mode: bool = False,
    ) -> None:
        self._node_map = _prepare_nodes(nodes)
        self.nodes = nodes
        self.name = name
        self.parallel_mode = parallel_mode
        self._dependency_cache: dict[tuple[str, str], bool] = {}

    def run(self, ctx: InferenceContext) -> InferenceContext:
        working_ctx = InferenceContext(
            input_path=ctx.input_path,
            work_dir=ctx.work_dir,
            device=ctx.device,
            config=ctx.config,
            artifacts=deepcopy(ctx.artifacts),
        )
        pending = {node.node_id: node for node in self.nodes}
        completed: set[str] = set()
        running: dict[Future[InferenceContext], tuple[InferenceGraphNode, dict[str, Any]]] = {}
        artifact_owners: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max(len(self.nodes), 1)) as executor:
            while pending or running:
                ready_ids = [
                    node_id
                    for node_id, node in pending.items()
                    if all(dependency_id in completed for dependency_id in node.after)
                ]

                for node_id in ready_ids:
                    node = pending.pop(node_id)
                    step_ctx = InferenceContext(
                        input_path=working_ctx.input_path,
                        work_dir=working_ctx.work_dir,
                        device=working_ctx.device,
                        config=working_ctx.config,
                        artifacts=deepcopy(working_ctx.artifacts),
                    )
                    baseline = deepcopy(step_ctx.artifacts)
                    future = executor.submit(self._run_step, node.step, step_ctx)
                    running[future] = (node, baseline)

                if not running:
                    unresolved = {node.node_id: list(node.after) for node in pending.values()}
                    raise ValueError(
                        "Inference pipeline has unresolved step dependencies or a dependency cycle: "
                        f"{unresolved}"
                    )

                completed_futures, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    node, baseline = running.pop(future)
                    step_ctx = future.result()
                    self._merge_step_result(
                        target=working_ctx,
                        source=step_ctx,
                        baseline=baseline,
                        node=node,
                        artifact_owners=artifact_owners,
                    )

                    completed.add(node.node_id)

        ctx.artifacts = working_ctx.artifacts
        return ctx

    @staticmethod
    def _run_step(step: BaseInferenceStep, ctx: InferenceContext) -> InferenceContext:
        return step.run(ctx)

    def _merge_step_result(
        self,
        *,
        target: InferenceContext,
        source: InferenceContext,
        baseline: dict[str, Any],
        node: InferenceGraphNode,
        artifact_owners: dict[str, str],
    ) -> None:
        for key in self._detect_artifact_updates(baseline, source.artifacts):
            owner = artifact_owners.get(key)
            if (
                self.parallel_mode
                and owner is not None
                and owner != node.node_id
                and not self._depends_on_node(node.node_id, owner)
            ):
                raise ValueError(
                    f"Artifact '{key}' was already written by step '{owner}' and "
                    f"cannot be overwritten by '{node.node_id}'."
                )

            target.set_artifact(key, source.artifacts[key])
            artifact_owners[key] = node.node_id

    def _depends_on_node(self, node_id: str, dependency_id: str) -> bool:
        key = (node_id, dependency_id)
        if key in self._dependency_cache:
            return self._dependency_cache[key]

        if node_id == dependency_id:
            self._dependency_cache[key] = True
            return True

        node = self._node_map[node_id]
        result = any(self._depends_on_node(parent_id, dependency_id) for parent_id in node.after)
        self._dependency_cache[key] = result
        return result

    def _detect_artifact_updates(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> Iterable[str]:
        for key, value in current.items():
            if key not in baseline:
                yield key
                continue

            previous = baseline[key]
            if self._values_differ(previous, value):
                yield key

    @staticmethod
    def _values_differ(previous: Any, current: Any) -> bool:
        try:
            result = previous != current
        except Exception:
            return True

        return result if isinstance(result, bool) else True
