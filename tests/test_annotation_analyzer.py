import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from task_latent.tagging.annotations import AnnotationAnalyzer, load_annotations
from task_latent.tagging.ontology import (
    ResponseEffector,
    ResponseLaterality,
    StimulusDynamics,
    StimulusLaterality,
    StimulusModality,
    StimulusType,
)
from task_latent.tagging.tag import ConditionAnnotation


def annotation(
    task, name, atlas=(), stimulus_types=(), modality=StimulusModality.AUDITORY
):
    return ConditionAnnotation(
        name=name,
        task=task,
        cognitive_atlas=atlas,
        stimulus_type=tuple((term, 0.9, None) for term in stimulus_types),
        stimulus_modality=(modality, 0.9, None),
        stimulus_laterality=(StimulusLaterality.UNSPECIFIED, 0.7, None),
        stimulus_dynamics=(StimulusDynamics.CONTINUOUS, 0.8, None),
        response_effector=(ResponseEffector.NONE, 0.9, None),
        response_laterality=(ResponseLaterality.NONE, 0.9, None),
    )


def fixture():
    return {
        "A": {
            "shared": annotation(
                "A",
                "shared",
                ("x", "x"),
                (
                    StimulusType.LANGUAGE,
                    StimulusType.LANGUAGE,
                    StimulusType.SYMBOLIC_NUMERICAL,
                ),
            ),
            "second": annotation("A", "second", ("y",), (), StimulusModality.VISUAL),
        },
        "B": {
            "shared": annotation("B", "shared", ("x", "z"), (StimulusType.LANGUAGE,)),
            "other": annotation("B", "other", (), (), StimulusModality.UNSPECIFIED),
        },
        "Empty": {},
    }


class AnnotationAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.analyzer = AnnotationAnalyzer(self.data)

    def test_term_and_task_summaries_deduplicate_and_use_all_tasks(self):
        terms = self.analyzer.term_summary("cognitive_atlas").set_index("label")
        x = terms.loc["x"]
        self.assertEqual((x.condition_count, x.task_count), (2, 2))
        self.assertEqual(x.tasks, ["A", "B"])
        self.assertEqual(x.condition_fraction, 0.5)
        self.assertAlmostEqual(x.task_fraction, 2 / 3)
        self.assertEqual(list(terms.index), ["x", "y", "z"])
        language = (
            self.analyzer.term_summary("stimulus_type")
            .set_index("label")
            .loc["language"]
        )
        self.assertEqual(language.condition_count, 2)
        by_task = self.analyzer.task_summary("cognitive_atlas")
        self.assertEqual(
            list(zip(by_task.task, by_task.label)),
            [("A", "x"), ("A", "y"), ("B", "x"), ("B", "z")],
        )
        self.assertTrue((by_task.condition_count == 1).all())
        self.assertTrue((by_task.task_condition_fraction == 0.5).all())

    def test_condition_overlap_and_directional_fractions(self):
        table = self.analyzer.overlap("cognitive_atlas", "stimulus_type")
        row = table[
            (table.left_label == "x") & (table.right_label == "symbolic_numerical")
        ].iloc[0]
        self.assertEqual(
            (row.intersection_count, row.left_count, row.right_count, row.union_count),
            (1, 2, 1, 2),
        )
        self.assertEqual(
            (row.jaccard, row.left_fraction, row.right_fraction), (0.5, 0.5, 1.0)
        )
        self.assertEqual(len(table), 3)

    def test_task_overlap_includes_different_conditions(self):
        condition = self.analyzer.overlap("cognitive_atlas", "stimulus_modality")
        self.assertFalse(
            ((condition.left_label == "x") & (condition.right_label == "visual")).any()
        )
        task = self.analyzer.overlap(
            "cognitive_atlas", "stimulus_modality", level="task"
        )
        row = task[(task.left_label == "x") & (task.right_label == "visual")].iloc[0]
        self.assertEqual(
            (row.intersection_count, row.left_count, row.right_count, row.union_count),
            (1, 2, 1, 2),
        )
        self.assertEqual(row.jaccard, 0.5)

    def test_condition_identity_not_pooled_on_either_side(self):
        table = self.analyzer.overlap("condition", "cognitive_atlas")
        shared = table[(table.left_condition == "shared") & (table.right_label == "x")]
        self.assertEqual(set(shared.left_task), {"A", "B"})
        self.assertTrue((shared.left_count == 1).all())
        reverse = self.analyzer.overlap("cognitive_atlas", "condition")
        shared = reverse[
            (reverse.right_condition == "shared") & (reverse.left_label == "x")
        ]
        self.assertEqual(set(shared.right_task), {"A", "B"})
        task = self.analyzer.overlap("condition", "cognitive_atlas", level="task")
        self.assertEqual(
            len(task[(task.left_condition == "shared") & (task.right_label == "x")]), 2
        )

    def test_coverage_distinguishes_empty_unspecified_and_absence(self):
        coverage = self.analyzer.coverage_summary().set_index("namespace")
        self.assertEqual(coverage.loc["cognitive_atlas", "missing_count"], 1)
        self.assertEqual(coverage.loc["stimulus_type", "missing_count"], 2)
        self.assertEqual(coverage.loc["stimulus_modality", "unspecified_count"], 1)
        self.assertEqual(coverage.loc["response_effector", "absence_count"], 4)
        self.assertEqual(coverage.loc["response_laterality", "absence_fraction"], 1.0)
        self.assertTrue(
            math.isnan(coverage.loc["cognitive_atlas", "unspecified_count"])
        )

    def test_json_round_trip_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        task: {
                            name: asdict(value) for name, value in conditions.items()
                        }
                        for task, conditions in self.data.items()
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_annotations(path)
            self.assertIsInstance(loaded["A"]["shared"].stimulus_type, list)
            analyzer = AnnotationAnalyzer(loaded)
            self.assertTrue(
                analyzer.term_summary("stimulus_type").equals(
                    self.analyzer.term_summary("stimulus_type")
                )
            )
            loaded["A"]["shared"].cognitive_atlas.append("new")
            self.assertNotIn("new", set(analyzer.term_summary("cognitive_atlas").label))

    def test_empty_inputs_produce_stable_schemas_and_nan_fractions(self):
        empty = AnnotationAnalyzer({"Empty": {}})
        self.assertEqual(
            list(empty.term_summary("cognitive_atlas").columns),
            [
                "namespace",
                "label",
                "condition_count",
                "task_count",
                "condition_fraction",
                "task_fraction",
                "tasks",
            ],
        )
        self.assertTrue(empty.term_summary("cognitive_atlas").empty)
        self.assertTrue(empty.task_summary("cognitive_atlas").empty)
        self.assertTrue(empty.overlap("condition", "cognitive_atlas").empty)
        coverage = empty.coverage_summary().set_index("namespace")
        self.assertEqual(coverage.loc["stimulus_type", "missing_count"], 0)
        self.assertTrue(math.isnan(coverage.loc["stimulus_type", "missing_fraction"]))
        self.assertTrue(AnnotationAnalyzer({}).term_summary("cognitive_atlas").empty)

    def test_invalid_inputs_have_context(self):
        original = self.data["A"]["shared"]
        for value, message in (
            (replace(original, name="different"), "A/shared: annotation identity"),
            (
                replace(original, stimulus_modality=("visual",)),
                "A/shared/stimulus_modality",
            ),
            (
                replace(original, stimulus_modality=("not_a_modality", 0.9, None)),
                "A/shared/stimulus_modality",
            ),
            (replace(original, cognitive_atlas=("",)), "A/shared/cognitive_atlas"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                AnnotationAnalyzer({"A": {"shared": value}})
        with self.assertRaisesRegex(TypeError, "A/shared/stimulus_modality"):
            AnnotationAnalyzer(
                {
                    "A": {
                        "shared": replace(
                            original,
                            stimulus_modality=(StimulusType.LANGUAGE, 0.9, None),
                        )
                    }
                }
            )
        with self.assertRaises(ValueError):
            self.analyzer.term_summary("condition")
        with self.assertRaises(ValueError):
            self.analyzer.overlap("cognitive_atlas", "cognitive_atlas")
        with self.assertRaises(ValueError):
            self.analyzer.overlap("condition", "cognitive_atlas", level="invalid")


if __name__ == "__main__":
    unittest.main()
