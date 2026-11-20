import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from task_latent.tagging.ontology import (
    ResponseEffector,
    ResponseLaterality,
    StimulusType,
)
from task_latent.tagging.tag import (
    ConditionAnnotation,
    AnnotationResponse,
    annotate_conditions,
    parse_cognitive_atlas_labels,
    prepare_conditions,
    tag_condition,
)


def item(term, confidence=0.9, comment=None):
    return {
        "ontology_term": term,
        "annotation_confidence": confidence,
        "annotation_comment": comment,
    }


VALID_RESPONSE = {
    "stimulus_type": [item("language")],
    "stimulus_modality": item("auditory"),
    "stimulus_laterality": item("unspecified"),
    "stimulus_dynamics": item("temporally_sequenced"),
    "response_effector": item("hand"),
    "response_laterality": item("right"),
}

METADATA = {
    "Task": {
        "keep": True,
        "condition": ["first", "second"],
        "description": "A task description.",
        "condition_descriptions": {
            "first": "First condition.",
            "second": "Second condition.",
        },
        "condition_tags": {"first": "['tag_a', 'tag_b']", "second": ["tag_c"]},
    },
    "Excluded": {"keep": False},
}


def response(value):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))


class TaggingTests(unittest.TestCase):
    def test_schema_rejects_bad_enums_missing_and_extra_fields(self):
        for field in VALID_RESPONSE:
            invalid = dict(VALID_RESPONSE)
            invalid[field] = (
                [item("not_a_term")]
                if isinstance(invalid[field], list)
                else item("not_a_term")
            )
            with self.subTest(field=field), self.assertRaises(ValidationError):
                AnnotationResponse.model_validate(invalid)
        for invalid in (
            {
                key: value
                for key, value in VALID_RESPONSE.items()
                if key != "stimulus_modality"
            },
            {**VALID_RESPONSE, "name": "invented"},
        ):
            with self.assertRaises(ValidationError):
                AnnotationResponse.model_validate(invalid)
        with self.assertRaises(ValidationError):
            AnnotationResponse.model_validate_json("not JSON")
        for invalid_item in (
            item("language"),
            item("auditory", confidence=1.1),
            {"ontology_term": "auditory"},
            {**item("auditory"), "extra": True},
        ):
            with self.assertRaises(ValidationError):
                AnnotationResponse.model_validate(
                    {**VALID_RESPONSE, "stimulus_modality": invalid_item}
                )
        validated = AnnotationResponse.model_validate(VALID_RESPONSE)
        self.assertEqual(
            validated.stimulus_type[0].ontology_term, StimulusType.LANGUAGE
        )
        self.assertEqual(validated.stimulus_modality.annotation_confidence, 0.9)

    @patch("task_latent.tagging.tag.chat")
    def test_tag_condition_preserves_identity_and_uses_schema(self, mock_chat):
        mock_chat.return_value = response(VALID_RESPONSE)
        result = tag_condition("first", "Task", "Condition.", "Task.", ["tag_a"])
        self.assertIsInstance(result, ConditionAnnotation)
        self.assertEqual(
            (result.name, result.task, result.cognitive_atlas),
            ("first", "Task", ("tag_a",)),
        )
        self.assertEqual(result.stimulus_type, (StimulusType.LANGUAGE,))
        self.assertEqual(result.response_effector, ResponseEffector.HAND)
        self.assertEqual(result.response_laterality, ResponseLaterality.RIGHT)
        args = mock_chat.call_args.kwargs
        self.assertEqual(args["format"], AnnotationResponse.model_json_schema())
        evidence = json.loads(
            args["messages"][0]["content"].split("Evidence and allowed terms:\n", 1)[1]
        )
        self.assertEqual(evidence["cognitive_atlas_terms"], ["tag_a"])
        self.assertIn("hand", evidence["ontology"]["response_effector"]["terms"])
        self.assertIn("unspecified", evidence["ontology"]["stimulus_type"]["terms"])
        prompt = args["messages"][0]["content"]
        self.assertIn('"annotation_confidence": 0.9', prompt)
        self.assertIn('"annotation_comment": null', prompt)

    def test_tags_and_input_validation(self):
        self.assertEqual(parse_cognitive_atlas_labels("['a', 'b']"), ["a", "b"])
        self.assertEqual(parse_cognitive_atlas_labels(["a"]), ["a"])
        self.assertEqual(parse_cognitive_atlas_labels(None), [])
        for bad in ("invalid", "[1]", {"a": 1}, "'single tag'"):
            with self.assertRaises(ValueError):
                parse_cognitive_atlas_labels(bad)
        self.assertEqual(len(prepare_conditions(METADATA)), 2)
        with self.assertRaises(ValueError):
            prepare_conditions(METADATA, ["Excluded"])
        with self.assertRaises(ValueError):
            prepare_conditions(
                {"Task": {**METADATA["Task"], "condition_descriptions": {}}}
            )

    @patch("task_latent.tagging.tag.chat")
    def test_checkpoint_and_existing_output_protection(self, mock_chat):
        mock_chat.side_effect = [
            response(VALID_RESPONSE),
            RuntimeError("network error"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.json"
            with self.assertRaisesRegex(RuntimeError, "Task/second"):
                annotate_conditions(METADATA, output)
            data = json.loads(output.read_text())
            self.assertEqual(list(data["Task"]), ["first"])
            self.assertEqual(
                data["Task"]["first"]["cognitive_atlas"], ["tag_a", "tag_b"]
            )
            self.assertEqual(data["Task"]["first"]["response_laterality"], "right")
            with self.assertRaises(FileExistsError):
                annotate_conditions(METADATA, output)
            mock_chat.side_effect = None
            mock_chat.return_value = response(VALID_RESPONSE)
            result = annotate_conditions(METADATA, output, overwrite=True)
            self.assertEqual(set(result["Task"]), {"first", "second"})
            self.assertNotIn("Excluded", json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
