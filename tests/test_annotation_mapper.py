import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from task_latent.tagging.annotations import (
    AnnotationAnalyzer,
    AnnotationMapper,
    load_annotations,
)
from task_latent.tagging.ontology import (
    ResponseEffector,
    ResponseLaterality,
    StimulusDynamics,
    StimulusLaterality,
    StimulusModality,
    StimulusType,
)
from task_latent.tagging.tag import ConditionAnnotation


def annotation(task, name, atlas=(), types=(), modality=StimulusModality.AUDITORY):
    return ConditionAnnotation(
        name=name,
        task=task,
        cognitive_atlas=atlas,
        stimulus_type=tuple((term, 0.9, None) for term in types),
        stimulus_modality=(modality, 0.9, None),
        stimulus_laterality=(StimulusLaterality.UNSPECIFIED, 0.8, None),
        stimulus_dynamics=(StimulusDynamics.NONE, 0.8, None),
        response_effector=(ResponseEffector.NONE, 0.9, None),
        response_laterality=(ResponseLaterality.NONE, 0.9, None),
    )


def fixture():
    return {
        "B": {
            "shared": annotation(
                "B", "shared", ("y",), (StimulusType.LANGUAGE,), StimulusModality.VISUAL
            ),
            "empty": annotation("B", "empty", modality=StimulusModality.NONE),
        },
        "A": {
            "shared": annotation(
                "A",
                "shared",
                ("x", "x", "z"),
                (
                    StimulusType.LANGUAGE,
                    StimulusType.SYMBOLIC_NUMERICAL,
                    StimulusType.LANGUAGE,
                ),
            ),
        },
        "Empty": {},
    }


class AnnotationMapperTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def test_single_labels_and_task_qualified_identities_match_analyzer(self):
        namespaces = ("cognitive_atlas", "stimulus_type", "stimulus_modality")
        mapping = AnnotationMapper(self.data, namespaces).map_combinations(order=1)
        self.assertEqual(
            list(mapping), [("A", "shared"), ("B", "empty"), ("B", "shared")]
        )
        self.assertEqual(
            mapping[("A", "shared")],
            (
                (("cognitive_atlas", "x"),),
                (("cognitive_atlas", "z"),),
                (("stimulus_modality", "auditory"),),
                (("stimulus_type", "language"),),
                (("stimulus_type", "symbolic_numerical"),),
            ),
        )
        self.assertIn((("stimulus_modality", "visual"),), mapping[("B", "shared")])
        matrix = AnnotationAnalyzer(self.data).indicator_matrix(namespaces)
        for identity, features in mapping.items():
            active = {
                (namespace, label)
                for namespace, label in matrix.columns
                if matrix.at[identity, (namespace, label)] == 1
            }
            self.assertEqual({feature[0] for feature in features}, active)

    def test_combinations_are_local_products_not_global_products(self):
        mapper = AnnotationMapper(
            self.data, ("cognitive_atlas", "stimulus_type", "stimulus_modality")
        )
        result = mapper.map_combinations(order=3)
        self.assertEqual(len(result[("A", "shared")]), 4)
        self.assertEqual(len(result[("B", "shared")]), 1)
        self.assertEqual(result[("B", "empty")], ())
        all_features = {feature for features in result.values() for feature in features}
        self.assertNotIn(
            (
                ("cognitive_atlas", "x"),
                ("stimulus_type", "language"),
                ("stimulus_modality", "visual"),
            ),
            all_features,
        )
        self.assertIn(
            (
                ("cognitive_atlas", "z"),
                ("stimulus_type", "symbolic_numerical"),
                ("stimulus_modality", "auditory"),
            ),
            all_features,
        )

    def test_collapse_is_namespace_scoped_simultaneous_and_deduplicated(self):
        mapper = AnnotationMapper(
            self.data,
            ("cognitive_atlas", "stimulus_type"),
            collapse={
                "cognitive_atlas": {
                    "x": "pooled",
                    "z": "pooled",
                    "pooled": "not_chained",
                },
                "stimulus_type": {
                    StimulusType.LANGUAGE: "verbal",
                    "symbolic_numerical": "verbal",
                },
            },
        )
        self.assertEqual(
            mapper.map_combinations(order=1)[("A", "shared")],
            ((("cognitive_atlas", "pooled"),), (("stimulus_type", "verbal"),)),
        )
        self.assertEqual(
            mapper.map_combinations(order=2)[("A", "shared")],
            ((("cognitive_atlas", "pooled"), ("stimulus_type", "verbal")),),
        )

    def test_all_pairs_and_triples_and_missing_namespace(self):
        namespaces = (
            "cognitive_atlas",
            "stimulus_type",
            "stimulus_modality",
            "response_effector",
        )
        mapper = AnnotationMapper(self.data, namespaces)
        pairs = mapper.map_combinations(order=2)
        triples = mapper.map_combinations(order=3)
        self.assertEqual(len(pairs[("A", "shared")]), 13)
        self.assertEqual(len(triples[("A", "shared")]), 12)
        self.assertEqual(
            pairs[("B", "empty")],
            (
                (
                    ("stimulus_modality", "none"),
                    ("response_effector", "no_overt_response"),
                ),
            ),
        )
        self.assertEqual(triples[("B", "empty")], ())
        self.assertFalse(
            any(
                len(feature) != 2 for features in pairs.values() for feature in features
            )
        )
        self.assertFalse(
            any(
                len(feature) != 3
                for features in triples.values()
                for feature in features
            )
        )

    def test_pair_restriction_normalizes_order_and_retains_empty_conditions(self):
        mapper = AnnotationMapper(
            self.data, ("cognitive_atlas", "stimulus_type", "stimulus_modality")
        )
        pairs = mapper.map_combinations(
            order=2, namespace_pairs=[("stimulus_modality", "stimulus_type")]
        )
        self.assertEqual(
            pairs[("A", "shared")],
            (
                (("stimulus_type", "language"), ("stimulus_modality", "auditory")),
                (
                    ("stimulus_type", "symbolic_numerical"),
                    ("stimulus_modality", "auditory"),
                ),
            ),
        )
        self.assertEqual(pairs[("B", "empty")], ())
        self.assertTrue(
            all(
                not values
                for values in mapper.map_combinations(
                    order=2, namespace_pairs=[]
                ).values()
            )
        )

    def test_min_count_uses_distinct_conditions_after_collapsing(self):
        mapper = AnnotationMapper(
            self.data,
            ("cognitive_atlas", "stimulus_type", "response_effector"),
            collapse={"cognitive_atlas": {"x": "common", "z": "common", "y": "common"}},
        )
        feature = (("cognitive_atlas", "common"), ("stimulus_type", "language"))
        pair = [("cognitive_atlas", "stimulus_type")]
        surviving = mapper.map_combinations(order=2, min_count=2, namespace_pairs=pair)
        self.assertEqual(surviving[("A", "shared")], (feature,))
        self.assertEqual(surviving[("B", "shared")], (feature,))
        self.assertEqual(surviving[("B", "empty")], ())
        self.assertTrue(
            all(
                not features
                for features in mapper.map_combinations(
                    order=2, min_count=3, namespace_pairs=pair
                ).values()
            )
        )
        singles = mapper.map_combinations(order=1, min_count=2)
        self.assertIn((("stimulus_type", "language"),), singles[("A", "shared")])
        self.assertNotIn(
            (("stimulus_type", "symbolic_numerical"),), singles[("A", "shared")]
        )
        triples = mapper.map_combinations(order=3, min_count=2)
        self.assertEqual(len(triples[("A", "shared")]), 1)
        self.assertEqual(triples[("A", "shared")], triples[("B", "shared")])

    def test_invalid_interaction_options(self):
        mapper = AnnotationMapper(
            self.data, ("cognitive_atlas", "stimulus_type", "stimulus_modality")
        )
        for kwargs in (
            {"order": 0},
            {"order": 4},
            {"min_count": 0},
            {"order": 1, "namespace_pairs": []},
            {"order": 3, "namespace_pairs": []},
            {"order": 2, "namespace_pairs": [("cognitive_atlas",)]},
            {"order": 2, "namespace_pairs": [("cognitive_atlas", "cognitive_atlas")]},
            {"order": 2, "namespace_pairs": [("cognitive_atlas", "response_effector")]},
            {
                "order": 2,
                "namespace_pairs": [
                    ("cognitive_atlas", "stimulus_type"),
                    ("stimulus_type", "cognitive_atlas"),
                ],
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mapper.map_combinations(**cast(Any, kwargs))
        with self.assertRaises(ValueError):
            AnnotationMapper(self.data, ("cognitive_atlas",)).map_combinations(order=2)
        for kwargs in (
            {"order": True},
            {"order": 2.0},
            {"min_count": False},
            {"min_count": 1.5},
            {"order": 2, "namespace_pairs": "invalid"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                mapper.map_combinations(**cast(Any, kwargs))

    def test_return_counts_match_retained_mapping_for_all_orders(self):
        mapper = AnnotationMapper(
            self.data,
            ("cognitive_atlas", "stimulus_type", "response_effector"),
            collapse={"cognitive_atlas": {"x": "common", "z": "common", "y": "common"}},
        )
        for order in (1, 2, 3):
            with self.subTest(order=order):
                mapping, counts = mapper.map_combinations(
                    order=cast(Any, order), min_count=2, return_counts=True
                )
                self.assertEqual(
                    mapping,
                    mapper.map_combinations(order=cast(Any, order), min_count=2),
                )
                expected = {
                    feature: sum(feature in features for features in mapping.values())
                    for features in mapping.values()
                    for feature in features
                }
                self.assertEqual(counts, expected)
                self.assertEqual(list(counts), sorted(counts))
                self.assertTrue(all(count >= 2 for count in counts.values()))

        pair = (("cognitive_atlas", "common"), ("stimulus_type", "language"))
        mapping, counts = mapper.map_combinations(
            order=2,
            min_count=2,
            namespace_pairs=[("stimulus_type", "cognitive_atlas")],
            return_counts=True,
        )
        self.assertEqual(counts, {pair: 2})
        self.assertEqual(mapping[("A", "shared")], (pair,))
        self.assertEqual(mapping[("B", "shared")], (pair,))
        _, singles = mapper.map_combinations(order=1, return_counts=True)
        self.assertEqual(singles[(("cognitive_atlas", "common"),)], 2)
        self.assertEqual(singles[(("stimulus_type", "language"),)], 2)
        self.assertEqual(singles[(("stimulus_type", "symbolic_numerical"),)], 1)

    def test_return_counts_empty_and_excluded(self):
        self.assertEqual(
            AnnotationMapper({}).map_combinations(return_counts=True), ({}, {})
        )
        mapper = AnnotationMapper(
            self.data,
            ("stimulus_modality", "response_effector"),
            exclude_none=("stimulus_modality",),
        )
        mapping, counts = mapper.map_combinations(order=2, return_counts=True)
        self.assertEqual(mapping[("B", "empty")], ())
        self.assertFalse(
            any(("stimulus_modality", "none") in feature for feature in counts)
        )
        self.assertEqual(
            mapper.map_combinations(order=2, namespace_pairs=[], return_counts=True),
            ({identity: () for identity in mapping}, {}),
        )
        self.assertEqual(
            mapper.map_combinations(order=2, min_count=10, return_counts=True),
            ({identity: () for identity in mapping}, {}),
        )

    def test_exclude_literal_none_only_and_keep_empty_conditions(self):
        namespaces = ("stimulus_modality", "response_effector", "stimulus_laterality")
        mapper = AnnotationMapper(
            self.data,
            namespaces,
            exclude_none=("stimulus_modality", "response_effector"),
        )
        empty = mapper.map_combinations(order=1)[("B", "empty")]
        self.assertEqual(
            empty,
            (
                (("response_effector", "no_overt_response"),),
                (("stimulus_laterality", "unspecified"),),
            ),
        )
        self.assertEqual(mapper.map_combinations(order=3)[("B", "empty")], ())
        self.assertEqual(
            AnnotationMapper(
                self.data, ("stimulus_modality",), exclude_none=("stimulus_modality",)
            ).map_combinations(order=1)[("B", "empty")],
            (),
        )
        self.assertEqual(
            AnnotationMapper(self.data, ("cognitive_atlas",)).map_combinations(order=1)[
                ("B", "empty")
            ],
            (),
        )

    def test_exclusion_precedes_collapse_and_is_namespace_specific(self):
        mapper = AnnotationMapper(
            self.data,
            ("stimulus_modality", "response_laterality"),
            collapse={"stimulus_modality": {"none": "absent", "auditory": "sensory"}},
            exclude_none=("stimulus_modality",),
        )
        self.assertEqual(
            mapper.map_combinations(order=1)[("B", "empty")],
            ((("response_laterality", "none"),),),
        )
        self.assertEqual(mapper.map_combinations(order=2)[("B", "empty")], ())
        self.assertEqual(
            mapper.map_combinations(order=2)[("A", "shared")],
            ((("stimulus_modality", "sensory"), ("response_laterality", "none")),),
        )

    def test_json_round_trip_and_snapshot(self):
        atlas = ["x"]
        collapses = {"cognitive_atlas": {"x": "group"}}
        data = {"A": {"one": annotation("A", "one", atlas)}}
        mapper = AnnotationMapper(data, ("cognitive_atlas",), collapse=collapses)
        atlas.append("later")
        collapses["cognitive_atlas"]["x"] = "changed"
        self.assertEqual(
            mapper.map_combinations(order=1)[("A", "one")],
            ((("cognitive_atlas", "group"),),),
        )
        result = mapper.map_combinations(order=1)
        result.clear()
        self.assertTrue(mapper.map_combinations(order=1))
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
            self.assertEqual(
                AnnotationMapper(loaded).map_combinations(order=1),
                AnnotationMapper(self.data).map_combinations(order=1),
            )
            self.assertEqual(
                AnnotationMapper(
                    loaded, ("stimulus_type", "stimulus_modality")
                ).map_combinations(order=2),
                AnnotationMapper(
                    self.data, ("stimulus_type", "stimulus_modality")
                ).map_combinations(order=2),
            )

    def test_empty_inputs_and_invalid_configuration(self):
        self.assertEqual(AnnotationMapper({}).map_combinations(order=1), {})
        self.assertEqual(AnnotationMapper({"A": {}}).map_combinations(), {})
        for kwargs in (
            {"namespaces": ()},
            {"namespaces": ("bad",)},
            {"namespaces": ("stimulus_type", "stimulus_type")},
            {"namespaces": ("cognitive_atlas",), "exclude_none": ("cognitive_atlas",)},
            {"namespaces": ("stimulus_type",), "exclude_none": ("stimulus_modality",)},
            {"namespaces": ("stimulus_type",), "collapse": {"stimulus_modality": {}}},
            {
                "namespaces": ("stimulus_type",),
                "collapse": {"stimulus_type": {"bad": "group"}},
            },
            {
                "namespaces": ("stimulus_type",),
                "collapse": {"stimulus_type": {"language": " "}},
            },
            {
                "namespaces": ("stimulus_type",),
                "exclude_none": ("stimulus_type",),
                "collapse": {"stimulus_type": {"language": "none"}},
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AnnotationMapper(self.data, **cast(Any, kwargs))
        for kwargs in (
            {"namespaces": "stimulus_type"},
            {"exclude_none": "stimulus_type"},
            {"collapse": []},
            {"collapse": {"stimulus_type": []}},
            {"collapse": {"stimulus_type": {StimulusModality.VISUAL: "group"}}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                AnnotationMapper(self.data, **cast(Any, kwargs))

    def test_invalid_annotation_identity_and_wrong_enum(self):
        original = self.data["A"]["shared"]
        with self.assertRaisesRegex(ValueError, "A/shared: annotation identity"):
            AnnotationMapper({"A": {"shared": replace(original, name="other")}})
        with self.assertRaisesRegex(TypeError, "A/shared/stimulus_modality"):
            AnnotationMapper(
                {
                    "A": {
                        "shared": replace(
                            original,
                            stimulus_modality=(StimulusType.LANGUAGE, 0.5, None),
                        )
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
