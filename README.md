# Condition tagging

The tagging pipeline uses the task descriptions, condition descriptions and
cognitive atlas tags in `task_latent/ibc_conditions.json`. It requires the
`task-coding` dependency group, a running Ollama server, and the
`qwen3.8:27b` model. For example:

```sh
uv sync --group task-coding
python -m task_latent.tagging.tag --task ArchiStandard
```

Omit `--task` to annotate every kept task; repeat it to select multiple tasks.
Use `--help` to select another input, output or model. The pipeline generates
one `ConditionAnnotation` per condition and checkpoints its serialized fields
after every successful response to `scratch/ibc_condition_annotations.json`.
It does not overwrite existing output unless `--overwrite` is specified; that
option starts a new run rather than resuming an existing checkpoint.

Pydantic validates the model response against the ontology enums, but valid
labels do not guarantee scientific accuracy. Review both the source descriptions
and generated annotations. The current ontology serializes the generic `HAND`
and `FOOT` effector categories as `left_hand` and `left_foot`; consult
`response_laterality` for the actual side.

## Analyzing annotations

The analyzer uses pandas and the `task-coding` dependency group. It reads a
snapshot of the annotation dictionary and does not require a running Ollama
server. For example:

```python
from task_latent.tagging.annotations import AnnotationAnalyzer, load_annotations

analyzer = AnnotationAnalyzer(load_annotations())
atlas = analyzer.term_summary("cognitive_atlas")
modalities_by_task = analyzer.task_summary("stimulus_modality")
condition_overlap = analyzer.overlap("stimulus_modality", "stimulus_type")
task_overlap = analyzer.overlap("cognitive_atlas", "stimulus_type", level="task")
condition_tags = analyzer.overlap("condition", "cognitive_atlas")
coverage = analyzer.coverage_summary()
```

All methods return pandas DataFrames. The available annotation namespaces are
`cognitive_atlas`, `stimulus_type`, `stimulus_modality`, `stimulus_laterality`,
`stimulus_dynamics`, `response_effector`, `response_laterality`, and
`response_structure`. `condition` is also available for overlap comparisons.
Each condition is identified by its task and condition keys, even when another
task uses the same condition name. Duplicate labels within one condition count
only once. Overall condition fractions use all input conditions; overall task
fractions include empty tasks. Task summaries use the conditions within the
corresponding task.

Overlap is **co-occurrence**, not text matching: condition-level overlap means
both labels occur on the same condition, while task-level overlap means they
occur somewhere in the same task, possibly in different conditions. Counts,
Jaccard similarity, and directional overlap fractions are reported for
nonzero intersections. Coverage separately reports missing annotations,
`unspecified`, and explicit absence (`none` or `no_overt_response`).
Confidence scores and comments are not included in the statistics.

## Preparing annotation mappings

`AnnotationMapper` maps each grouped `(task, condition)` to one or more
namespace-qualified features. It uses the same annotation snapshot as the
analyzer; it does not create the time-series design matrix itself.

```python
from task_latent.tagging.annotations import AnnotationMapper, load_annotations

mapper = AnnotationMapper(
	load_annotations(),
	("stimulus_modality", "stimulus_laterality", "cognitive_atlas"),
	collapse={"stimulus_modality": {"visual": "sensory", "auditory": "sensory"}},
	exclude_none=("stimulus_modality",),
)
individual = mapper.map_combinations(order=1)
all_pairs = mapper.map_combinations(order=2, min_count=2)
selected_pairs = mapper.map_combinations(
	order=2, namespace_pairs=[("stimulus_modality", "stimulus_laterality")]
)
three_way = mapper.map_combinations(order=3)
mapping, counts = mapper.map_combinations(order=2, min_count=2, return_counts=True)
```

Every feature is a tuple of `(namespace, label)` pairs. `order` selects exactly
one, two, or three distinct namespaces per feature; by default all namespace
subsets of that size are crossed. For two-way crosses, `namespace_pairs`
restricts the selected pairs. Each feature must occur in at least `min_count`
distinct task-condition pairs (default 1), counted after collapsing and
exclusion; labels in different conditions are never crossed. Multi-valued
labels can produce multiple features, so event
rows must be expanded or encoded using `(task, trial_type)` rather than
replaced one-to-one. Conditions without matching labels remain as empty tuples.
With `return_counts=True`, the second returned dictionary maps each retained
feature to its number of distinct task-condition pairs; filtered-out features
are omitted. The default return value is the mapping alone.
Collapses are namespace-specific and applied once. `exclude_none` removes only
the literal ontology label `none`, before collapsing; `no_overt_response` and
`unspecified` remain informative features.

