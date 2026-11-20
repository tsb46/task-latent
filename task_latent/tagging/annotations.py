"""
Utilities for handling condition annotations and ontology mappings.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from enum import Enum
from itertools import combinations, product
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd

from .ontology import (
    ResponseEffector,
    ResponseLaterality,
    StimulusDynamics,
    StimulusLaterality,
    StimulusModality,
    StimulusType,
)
from .tag import DEFAULT_OUTPUT, ConditionAnnotation

ONTOLOGY_FIELDS: dict[str, type[Enum]] = {
    "stimulus_type": StimulusType,
    "stimulus_modality": StimulusModality,
    "stimulus_laterality": StimulusLaterality,
    "stimulus_dynamics": StimulusDynamics,
    "response_effector": ResponseEffector,
    "response_laterality": ResponseLaterality,
}
ANNOTATION_NAMESPACES = ("cognitive_atlas", *ONTOLOGY_FIELDS)

ConditionId = tuple[str, str]
Label = str | ConditionId
QualifiedLabel = tuple[str, str]
Feature = tuple[QualifiedLabel, ...]
FeatureMapping = dict[ConditionId, tuple[Feature, ...]]
FeatureCounts = dict[Feature, int]


def load_annotations(
    input_path: Path = DEFAULT_OUTPUT,
) -> dict[str, dict[str, ConditionAnnotation]]:
    """Load annotations from a JSON file.

    Args:
        input_path: Path to the JSON checkpoint. Defaults to the tagging
            pipeline's output path.

    Returns:
        A mapping from task names to condition labels and their corresponding
        ConditionAnnotation objects. Nested JSON lists and strings are not
        converted back into tuples or ontology enums.
    """
    with input_path.open(encoding="utf-8") as file:
        data = json.load(file)
    return {
        t: {c: ConditionAnnotation(**value) for c, value in entries.items()}
        for t, entries in data.items()
    }


def _ontology_label(
    item: object, enum_type: type[Enum], identity: ConditionId, field: str
) -> str:
    """Validate an ontology triple and return its string-valued term."""
    context = f"{identity[0]}/{identity[1]}/{field}"
    if not isinstance(item, (tuple, list)) or len(item) != 3:
        raise ValueError(f"{context}: expected a (term, confidence, comment) triple")
    term = item[0]
    if isinstance(term, Enum):
        if not isinstance(term, enum_type):
            raise TypeError(f"{context}: term belongs to the wrong ontology")
        term = term.value
    if not isinstance(term, str):
        raise TypeError(f"{context}: ontology term must be a string or enum")
    try:
        enum_type(term)
    except ValueError as exc:
        raise ValueError(f"{context}: unknown ontology term {term!r}") from exc
    return term


def _normalize_condition(
    annotation: ConditionAnnotation, identity: ConditionId
) -> dict[str, set[str]]:
    """Validate identity and terms, and snapshot labels without confidence."""
    task, condition = identity
    if annotation.task != task or annotation.name != condition:
        raise ValueError(
            f"{task}/{condition}: annotation identity "
            f"({annotation.task!r}, {annotation.name!r}) does not match keys"
        )
    atlas = annotation.cognitive_atlas
    if not isinstance(atlas, (list, tuple)) or not all(
        isinstance(label, str) and label.strip() for label in atlas
    ):
        raise ValueError(
            f"{task}/{condition}/cognitive_atlas: expected a collection of nonblank strings"
        )
    labels = {"cognitive_atlas": set(atlas)}
    for field, enum_type in ONTOLOGY_FIELDS.items():
        value = getattr(annotation, field)
        if field == "stimulus_type":
            if not isinstance(value, (tuple, list)):
                raise ValueError(
                    f"{task}/{condition}/{field}: expected a collection of annotation triples"
                )
            items = value
        else:
            items = (value,)
        labels[field] = {
            _ontology_label(item, enum_type, identity, field) for item in items
        }
    return labels


class AnnotationMapper:
    """Prepare task-specific conditions for annotation-based design matrices.

    The mapper snapshots annotations and ignores confidence and comments.
    Every feature is a tuple of ``(namespace, label)`` pairs, with one pair
    per namespace in the requested interaction.
    Conditions with no features remain in the returned mapping. These keys
    correspond to the loader's grouped ``(task, trial_type)`` values; because
    a condition can have multiple features, callers must expand or encode the
    assignments rather than use a one-to-one ``trial_type.replace``.

    Example:
        ``AnnotationMapper(annotations, ("stimulus_modality",),
        collapse={"stimulus_modality": {"auditory": "sensory",
                                        "visual": "sensory"}},
        exclude_none=("stimulus_modality",)).map_combinations(order=1)``
    """

    def __init__(
        self,
        annotations: Mapping[str, Mapping[str, ConditionAnnotation]],
        namespaces: Sequence[str] = ANNOTATION_NAMESPACES,
        *,
        collapse: Mapping[str, Mapping[str, str]] | None = None,
        exclude_none: Collection[str] = (),
    ) -> None:
        """Configure namespaces, optional label collapses, and exclusions.

        Args:
            annotations: Task -> grouped condition -> annotation mapping.
            namespaces: Nonempty, distinct ontology fields and/or
                ``cognitive_atlas``. Order determines combination pair order.
            collapse: Namespace -> source label -> output label. Replacements
                are simultaneous, not chained. Ontology sources must belong to
                that ontology; targets may be arbitrary nonblank strings.
            exclude_none: Selected ontology namespaces in which to omit the
                literal ``none`` label before collapsing. Cognitive Atlas
                is not an ontology namespace for this option.

        Raises:
            ValueError: If configuration or annotation values are invalid.
            TypeError: If ontology terms use an incorrect enum or type.
        """
        if isinstance(namespaces, str) or not isinstance(namespaces, Sequence):
            raise TypeError("namespaces must be a nonempty sequence of names")
        if not namespaces:
            raise ValueError("namespaces must not be empty")
        if any(not isinstance(namespace, str) for namespace in namespaces):
            raise ValueError("namespace names must be strings")
        if len(set(namespaces)) != len(namespaces):
            raise ValueError("Namespaces must not contain duplicates")
        for namespace in namespaces:
            if namespace not in ANNOTATION_NAMESPACES:
                raise ValueError(f"Unknown namespace {namespace!r}")
        self.namespaces = tuple(namespaces)

        if isinstance(exclude_none, str) or not isinstance(exclude_none, Collection):
            raise TypeError("exclude_none must be a collection of ontology names")
        for namespace in exclude_none:
            if namespace not in ONTOLOGY_FIELDS or namespace not in self.namespaces:
                raise ValueError(
                    f"exclude_none namespace {namespace!r} must be a selected ontology"
                )
        excluded = set(exclude_none)

        if collapse is not None and not isinstance(collapse, Mapping):
            raise TypeError("collapse must be a mapping of namespaces to mappings")
        replacements: dict[str, dict[str, str]] = {}
        for namespace, mapping in (collapse or {}).items():
            if namespace not in self.namespaces:
                raise ValueError(f"collapse namespace {namespace!r} is not selected")
            if not isinstance(mapping, Mapping):
                raise TypeError(f"collapse/{namespace}: expected a label mapping")
            replacements[namespace] = {}
            for source, target in mapping.items():
                if namespace in ONTOLOGY_FIELDS:
                    source = _ontology_label(
                        (source, None, None),
                        ONTOLOGY_FIELDS[namespace],
                        ("collapse", namespace),
                        namespace,
                    )
                elif not isinstance(source, str) or not source.strip():
                    raise ValueError(f"collapse/{namespace}: source must be nonblank")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError(f"collapse/{namespace}: target must be nonblank")
                if isinstance(target, Enum):
                    target = target.value
                if namespace in excluded and target == "none":
                    raise ValueError(f"collapse/{namespace}: target cannot be 'none'")
                replacements[namespace][source] = target

        self._labels: dict[ConditionId, dict[str, tuple[str, ...]]] = {}
        for task, conditions in sorted(annotations.items()):
            for condition, annotation in sorted(conditions.items()):
                identity = (task, condition)
                normalized = _normalize_condition(annotation, identity)
                self._labels[identity] = {}
                for namespace in self.namespaces:
                    terms = normalized[namespace]
                    if namespace in excluded:
                        terms.discard("none")
                    mapping = replacements.get(namespace, {})
                    self._labels[identity][namespace] = tuple(
                        sorted({mapping.get(term, term) for term in terms})
                    )

    @overload
    def map_combinations(
        self,
        order: Literal[1, 2, 3] = 1,
        *,
        min_count: int = 1,
        namespace_pairs: Sequence[tuple[str, str]] | None = None,
        return_counts: Literal[False] = False,
    ) -> FeatureMapping: ...

    @overload
    def map_combinations(
        self,
        order: Literal[1, 2, 3] = 1,
        *,
        min_count: int = 1,
        namespace_pairs: Sequence[tuple[str, str]] | None = None,
        return_counts: Literal[True],
    ) -> tuple[FeatureMapping, FeatureCounts]: ...

    @overload
    def map_combinations(
        self,
        order: Literal[1, 2, 3] = 1,
        *,
        min_count: int = 1,
        namespace_pairs: Sequence[tuple[str, str]] | None = None,
        return_counts: bool,
    ) -> FeatureMapping | tuple[FeatureMapping, FeatureCounts]: ...

    def map_combinations(
        self,
        order: Literal[1, 2, 3] = 1,
        *,
        min_count: int = 1,
        namespace_pairs: Sequence[tuple[str, str]] | None = None,
        return_counts: bool = False,
    ) -> FeatureMapping | tuple[FeatureMapping, FeatureCounts]:
        """Map conditions to observed labels or cross-namespace interactions.

        Args:
            order: Exact number of distinct namespaces per feature: 1 for
                individual labels, 2 for pairs, or 3 for three-way crosses.
                By default every subset of that size is crossed.
            min_count: Minimum number of distinct task-condition pairs in
                which a feature must occur, after exclusions and collapsing.
                Must be an integer of at least one.
            namespace_pairs: For order 2 only, optionally restrict crosses to
                these pairs of selected namespaces. Pair order is normalized
                to the order in ``namespaces``; an empty sequence selects none.
            return_counts: If true, also return counts for the retained
                features. Defaults to false, returning only the mapping.

        Returns:
            A deterministic mapping from each task-condition pair to its
            features. A feature has one namespace-qualified label per crossed
            namespace. Multiple labels in a namespace yield separate features;
            labels are crossed only within the same condition. Missing labels
            suppress only crosses involving that namespace. Conditions without
            retained features map to an empty tuple. If ``return_counts`` is
            true, return ``(mapping, counts)``, where counts maps each retained
            feature to its number of distinct task-condition pairs, after
            exclusion, collapse, pair selection and the ``min_count`` filter.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If an order, count, or namespace pair is invalid.
        """
        if type(order) is not int:
            raise TypeError("order must be an integer")
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3")
        if order > len(self.namespaces):
            raise ValueError("order cannot exceed the number of namespaces")
        if type(min_count) is not int:
            raise TypeError("min_count must be an integer")
        if min_count < 1:
            raise ValueError("min_count must be at least 1")

        crosses: tuple[tuple[str, ...], ...]
        if namespace_pairs is None:
            crosses = tuple(combinations(self.namespaces, order))
        else:
            if order != 2:
                raise ValueError("namespace_pairs can only be used with order=2")
            if isinstance(namespace_pairs, str) or not isinstance(
                namespace_pairs, Sequence
            ):
                raise TypeError("namespace_pairs must be a sequence of pairs")
            positions = {namespace: i for i, namespace in enumerate(self.namespaces)}
            selected: set[tuple[str, str]] = set()
            for pair in namespace_pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError("each namespace pair must be a two-element tuple")
                if any(
                    not isinstance(namespace, str) or namespace not in positions
                    for namespace in pair
                ):
                    raise ValueError(
                        f"namespace pair {pair!r} must contain selected namespaces"
                    )
                if pair[0] == pair[1]:
                    raise ValueError(
                        f"namespace pair {pair!r} must contain distinct namespaces"
                    )
                normalized = tuple(sorted(pair, key=positions.__getitem__))
                if normalized in selected:
                    raise ValueError(f"duplicate namespace pair {pair!r}")
                selected.add(normalized)
            crosses = tuple(
                pair for pair in combinations(self.namespaces, 2) if pair in selected
            )

        condition_features: dict[ConditionId, set[Feature]] = {}
        counts: Counter[Feature] = Counter()
        for identity, values in self._labels.items():
            features: set[Feature] = set()
            for cross in crosses:
                features.update(
                    product(
                        *(
                            tuple((namespace, label) for label in values[namespace])
                            for namespace in cross
                        )
                    )
                )
            condition_features[identity] = features
            counts.update(features)
        mapping = {
            identity: tuple(
                sorted(feature for feature in features if counts[feature] >= min_count)
            )
            for identity, features in condition_features.items()
        }
        if return_counts:
            return mapping, {
                feature: counts[feature]
                for feature in sorted(counts)
                if counts[feature] >= min_count
            }
        return mapping


class AnnotationAnalyzer:
    """Describe annotation prevalence and co-occurrence.

    Conditions are identified by their (task, condition) keys. Counts are of
    distinct conditions or tasks, not annotation occurrences. Input is copied
    into a normalized snapshot; later changes to the input do not affect it.
    Confidence and comments are not used in these descriptive statistics.
    """

    def __init__(self, annotations: dict[str, dict[str, ConditionAnnotation]]) -> None:
        """Create a normalized snapshot of the supplied annotations.

        Args:
            annotations: Mapping from task names to condition labels to
                annotations. Each annotation's task and name must match its
                mapping keys. Empty tasks are included in task denominators.

        Returns:
            None.

        Raises:
            ValueError: If an annotation's identity, Cognitive Atlas labels,
                ontology triple, or ontology term is invalid.
            TypeError: If an ontology term has an invalid type or enum class.
        """
        self._tasks = tuple(sorted(annotations))
        self._task_condition_counts = {
            task: len(conditions) for task, conditions in annotations.items()
        }
        self._conditions: set[ConditionId] = set()
        self._members: dict[str, dict[Label, set[ConditionId]]] = {
            namespace: defaultdict(set)
            for namespace in ("condition", *ANNOTATION_NAMESPACES)
        }
        for task, conditions in annotations.items():
            for condition, annotation in conditions.items():
                identity = (task, condition)
                labels = _normalize_condition(annotation, identity)
                self._conditions.add(identity)
                self._members["condition"][identity].add(identity)
                for namespace, terms in labels.items():
                    for label in terms:
                        self._members[namespace][label].add(identity)

    def _check_namespace(
        self, namespace: str, *, allow_condition: bool = False
    ) -> None:
        """Validate an annotation namespace.

        Args:
            namespace: Cognitive Atlas or ontology field name to validate.
            allow_condition: Whether the task-specific condition namespace is
                also accepted.

        Returns:
            None.

        Raises:
            ValueError: If the namespace is not accepted.
        """
        valid = (
            ("condition", *ANNOTATION_NAMESPACES)
            if allow_condition
            else ANNOTATION_NAMESPACES
        )
        if namespace not in valid:
            raise ValueError(
                f"Unknown namespace {namespace!r}; expected one of {', '.join(valid)}"
            )

    def term_summary(self, namespace: str) -> pd.DataFrame:
        """Summarize observed labels across all conditions and input tasks.

        Condition fractions use all conditions; task fractions use all input
        tasks, including tasks without conditions. The task list is sorted.

        Args:
            namespace: ``cognitive_atlas`` or an ontology field name.

        Returns:
            A DataFrame with one row per observed label and columns
            ``namespace``, ``label``, ``condition_count``, ``task_count``,
            ``condition_fraction``, ``task_fraction``, and ``tasks``. Counts
            refer to distinct conditions or tasks, and ``tasks`` is a sorted
            list of task names. Rows are sorted by label; if no labels are
            observed, the DataFrame is empty with the same columns.

        Raises:
            ValueError: If the namespace is not an annotation field.
        """
        self._check_namespace(namespace)
        columns = [
            "namespace",
            "label",
            "condition_count",
            "task_count",
            "condition_fraction",
            "task_fraction",
            "tasks",
        ]
        rows = []
        for label, conditions in sorted(self._members[namespace].items()):
            tasks = sorted({task for task, _ in conditions})
            rows.append(
                {
                    "namespace": namespace,
                    "label": label,
                    "condition_count": len(conditions),
                    "task_count": len(tasks),
                    "condition_fraction": len(conditions) / len(self._conditions),
                    "task_fraction": len(tasks) / len(self._tasks),
                    "tasks": tasks,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def task_summary(self, namespace: str) -> pd.DataFrame:
        """Summarize observed labels within each task.

        Args:
            namespace: ``cognitive_atlas`` or an ontology field name.

        Returns:
            A DataFrame with one row per observed task and label, containing
            ``task``, ``namespace``, ``label``, ``condition_count``, and
            ``task_condition_fraction``. The fraction divides the distinct
            condition count by the task's total conditions. Rows are sorted by
            task and label; an empty result retains these columns.

        Raises:
            ValueError: If the namespace is not an annotation field.
        """
        self._check_namespace(namespace)
        columns = [
            "task",
            "namespace",
            "label",
            "condition_count",
            "task_condition_fraction",
        ]
        rows = []
        for label, conditions in self._members[namespace].items():
            counts: dict[str, int] = defaultdict(int)
            for task, _ in conditions:
                counts[task] += 1
            for task, count in counts.items():
                rows.append(
                    {
                        "task": task,
                        "namespace": namespace,
                        "label": label,
                        "condition_count": count,
                        "task_condition_fraction": count
                        / self._task_condition_counts[task],
                    }
                )
        rows.sort(key=lambda row: (row["task"], row["label"]))
        return pd.DataFrame(rows, columns=columns)

    def overlap(
        self,
        left_namespace: str,
        right_namespace: str,
        *,
        level: Literal["condition", "task"] = "condition",
    ) -> pd.DataFrame:
        """Compare labels by shared conditions or by shared tasks.

        Only nonzero intersections are returned. At task level, two labels can
        overlap even when they occur in different conditions of the same task.
        A condition label is task-specific; its task and condition are included
        in the result rather than pooling identical names from different tasks.
        Directional fractions divide the intersection by the corresponding side.

        Args:
            left_namespace: ``condition``, ``cognitive_atlas``, or an ontology
                field name for the left side of the comparison.
            right_namespace: Namespace for the right side; it must differ from
                the left namespace.
            level: Count shared task-condition pairs with ``condition``, or
                shared tasks with ``task``. Defaults to ``condition``.

        Returns:
            A DataFrame with one row per label pair with a nonzero intersection.
            The ``left_namespace``, ``left_label``, ``right_namespace``, and
            ``right_label`` columns identify the pair. ``left_task`` and
            ``left_condition`` identify a task-specific condition on the left,
            and ``right_task`` and ``right_condition`` do so on the right;
            these columns are None for non-condition namespaces.

            ``intersection_count`` counts shared members; ``left_count`` and
            ``right_count`` count each side's members; and ``union_count``
            counts members present on either side. ``jaccard`` is intersection
            divided by union. ``left_fraction`` and ``right_fraction`` divide
            the intersection by the corresponding side's count. The membership
            unit is determined by ``level``. An empty result retains the same
            columns.

        Raises:
            ValueError: If a namespace or level is invalid, or the two
                namespaces are the same.
        """
        self._check_namespace(left_namespace, allow_condition=True)
        self._check_namespace(right_namespace, allow_condition=True)
        if left_namespace == right_namespace:
            raise ValueError("Overlap requires two distinct namespaces")
        if level not in ("condition", "task"):
            raise ValueError("level must be 'condition' or 'task'")
        columns = [
            "left_namespace",
            "left_label",
            "left_task",
            "left_condition",
            "right_namespace",
            "right_label",
            "right_task",
            "right_condition",
            "intersection_count",
            "left_count",
            "right_count",
            "union_count",
            "jaccard",
            "left_fraction",
            "right_fraction",
        ]

        def memberships(namespace: str) -> dict[Label, set]:
            """Get each label's condition or task membership at the chosen level.

            Args:
                namespace: Namespace whose label memberships are requested.

            Returns:
                A mapping from labels to sets of condition identities or task
                names, according to the enclosing overlap level.
            """
            if level == "condition":
                return self._members[namespace]
            return {
                label: {task for task, _ in conditions}
                for label, conditions in self._members[namespace].items()
            }

        left_members = memberships(left_namespace)
        right_members = memberships(right_namespace)
        rows = []
        for left_label, left_set in sorted(left_members.items()):
            for right_label, right_set in sorted(right_members.items()):
                intersection = len(left_set & right_set)
                if not intersection:
                    continue
                union = len(left_set | right_set)
                left_task, left_condition = (
                    left_label if left_namespace == "condition" else (None, None)
                )
                right_task, right_condition = (
                    right_label if right_namespace == "condition" else (None, None)
                )
                rows.append(
                    {
                        "left_namespace": left_namespace,
                        "left_label": left_condition
                        if left_namespace == "condition"
                        else left_label,
                        "left_task": left_task,
                        "left_condition": left_condition,
                        "right_namespace": right_namespace,
                        "right_label": right_condition
                        if right_namespace == "condition"
                        else right_label,
                        "right_task": right_task,
                        "right_condition": right_condition,
                        "intersection_count": intersection,
                        "left_count": len(left_set),
                        "right_count": len(right_set),
                        "union_count": union,
                        "jaccard": intersection / union,
                        "left_fraction": intersection / len(left_set),
                        "right_fraction": intersection / len(right_set),
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def coverage_summary(self) -> pd.DataFrame:
        """Report missing labels, unspecified labels, and explicit absence.

        Fractions use all conditions. For cognitive atlas, unspecified and
        explicit-absence columns are not applicable and are returned as NaN.

        Returns:
            A DataFrame with one row per annotation namespace. It contains
            ``namespace``, the total ``condition_count``, and counts and
            fractions for ``missing``, ``unspecified``, and ``absence``.
            Missing means no labels in the namespace; absence means ``none``
            or, for response effector, ``no_overt_response``. Each fraction
            divides its count by the total number of conditions and is NaN
            when that total is zero. Unspecified and absence counts and
            fractions are NaN for Cognitive Atlas.
        """
        columns = [
            "namespace",
            "condition_count",
            "missing_count",
            "missing_fraction",
            "unspecified_count",
            "unspecified_fraction",
            "absence_count",
            "absence_fraction",
        ]
        total = len(self._conditions)

        def fraction(count: int) -> float:
            """Calculate a fraction of all input conditions.

            Args:
                count: Number of conditions in the category.

            Returns:
                The proportion of all conditions, or NaN if there are none.
            """
            return count / total if total else float("nan")

        rows = []
        for namespace in ANNOTATION_NAMESPACES:
            members = self._members[namespace]
            covered: set[ConditionId] = set().union(*members.values())
            missing = total - len(covered)
            row = {
                "namespace": namespace,
                "condition_count": total,
                "missing_count": missing,
                "missing_fraction": fraction(missing),
                "unspecified_count": float("nan"),
                "unspecified_fraction": float("nan"),
                "absence_count": float("nan"),
                "absence_fraction": float("nan"),
            }
            if namespace in ONTOLOGY_FIELDS:
                unspecified = len(members.get("unspecified", set()))
                absence_label = (
                    "no_overt_response" if namespace == "response_effector" else "none"
                )
                absence = len(members.get(absence_label, set()))
                row.update(
                    {
                        "unspecified_count": unspecified,
                        "unspecified_fraction": fraction(unspecified),
                        "absence_count": absence,
                        "absence_fraction": fraction(absence),
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def indicator_matrix(
        self,
        namespaces: tuple[str, ...] | None = None,
        *,
        intercept: bool = False,
    ) -> pd.DataFrame:
        """Build a condition-by-annotation indicator matrix.

        Args:
            namespaces: Annotation namespaces to include. Defaults to all
                Cognitive Atlas and ontology namespaces. The condition namespace
                is not accepted. Only observed labels become columns.
            intercept: Whether to include a column of ones.

        Returns:
            A DataFrame indexed by (task, condition), with columns indexed by
            (namespace, label). Each annotation entry is one if the condition
            has that label and zero otherwise. Conditions without labels in a
            selected namespace remain in the matrix.

        Raises:
            ValueError: If a namespace is unknown or appears more than once.
        """
        if namespaces is None:
            namespaces = ANNOTATION_NAMESPACES

        if len(set(namespaces)) != len(namespaces):
            raise ValueError("Namespaces must not contain duplicates")

        for namespace in namespaces:
            self._check_namespace(namespace)

        conditions = sorted(self._conditions)
        row_positions = {condition: i for i, condition in enumerate(conditions)}

        features = [
            (namespace, label)
            for namespace in namespaces
            for label in sorted(self._members[namespace])
        ]
        if intercept:
            features.insert(0, ("intercept", "intercept"))

        values = np.zeros((len(conditions), len(features)), dtype=float)

        for column, (namespace, label) in enumerate(features):
            if namespace == "intercept":
                values[:, column] = 1.0
            else:
                for condition in self._members[namespace][label]:
                    values[row_positions[condition], column] = 1.0

        return pd.DataFrame(
            values,
            index=pd.MultiIndex.from_tuples(conditions, names=["task", "condition"]),
            columns=pd.MultiIndex.from_tuples(features, names=["namespace", "label"]),
        )

    def collinearity_diagnostics(
        self,
        namespaces: tuple[str, ...] | None = None,
        *,
        intercept: bool = False,
        near_rtol: float = 1e-7,
    ) -> dict[str, object]:
        """Diagnose linear dependencies in the annotation indicator matrix.

        Singular values are computed on the raw, unscaled matrix. Null-space
        directions describe combinations of columns, not necessarily pairs.
        An intercept and complete one-hot categorical fields normally introduce
        exact dependencies.

        Args:
            namespaces: Annotation namespaces to include. Defaults to all
                Cognitive Atlas and ontology namespaces.
            intercept: Whether to include an intercept column.
            near_rtol: Relative cutoff for reporting nearly dependent directions,
                as a fraction of the largest singular value. The numerical rank
                uses a separate machine-precision-based tolerance.

        Returns:
            A dictionary containing:

            - ``matrix``: The condition-by-annotation indicator DataFrame.
            - ``rank``: Numerical matrix rank.
            - ``n_columns``: Number of indicator columns, including the intercept.
            - ``rank_tolerance``: Absolute tolerance used to determine rank.
            - ``condition_number``: Largest divided by smallest singular value,
            or infinity when the matrix is column-rank deficient. NaN when
            there are no columns.
            - ``singular_values``: One value per column, including structural
            zeros when there are more columns than rows.
            - ``exact_dependencies``: DataFrame whose rows are numerical null-space
            directions and whose columns match the indicator matrix.
            - ``near_dependencies``: DataFrame of non-null singular directions at
            or below the relative cutoff, with the same columns.

        Raises:
            ValueError: If a namespace is invalid or duplicated, or ``near_rtol``
                is not between zero and one.
        """
        if not 0 <= near_rtol <= 1:
            raise ValueError("near_rtol must be between zero and one")

        matrix = self.indicator_matrix(namespaces, intercept=intercept)
        values = matrix.to_numpy()
        n_rows, n_columns = values.shape

        if n_rows == 0:
            singular_values = np.zeros(n_columns)
            right_vectors = np.eye(n_columns)
        elif n_columns == 0:
            singular_values = np.array([], dtype=float)
            right_vectors = np.empty((0, 0))
        else:
            # A wide matrix needs full Vh to retain its entire null space.
            _, observed_values, right_vectors = np.linalg.svd(
                values, full_matrices=n_rows < n_columns
            )
            singular_values = np.zeros(n_columns)
            singular_values[: len(observed_values)] = observed_values

        largest = singular_values[0] if n_columns else 0.0
        tolerance = max(n_rows, n_columns) * np.finfo(float).eps * largest
        exact = singular_values <= tolerance
        near = (~exact) & (singular_values <= near_rtol * largest)
        rank = int(np.count_nonzero(~exact))

        if n_columns == 0:
            condition_number = float("nan")
        elif rank < n_columns:
            condition_number = float("inf")
        else:
            condition_number = float(largest / singular_values[-1])

        return {
            "matrix": matrix,
            "rank": rank,
            "n_columns": n_columns,
            "rank_tolerance": tolerance,
            "condition_number": condition_number,
            "singular_values": pd.Series(singular_values, name="singular_value"),
            "exact_dependencies": pd.DataFrame(
                right_vectors[exact], columns=matrix.columns
            ),
            "near_dependencies": pd.DataFrame(
                right_vectors[near], columns=matrix.columns
            ),
        }
