"""
Ontology of stimulus and response characteristics in the IBC dataset
"""

from dataclasses import dataclass
from enum import Enum

#######################################################
######## Condition-Level Ontology Term Enums ##########
#######################################################


class StimulusModality(str, Enum):
    """Sensory channel used to present a task stimulus."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    AUDIOVISUAL = "audiovisual"
    NONE = "none"
    UNSPECIFIED = "unspecified"


class StimulusLaterality(str, Enum):
    """Laterality of the stimulus presentation."""

    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NONE = "none"
    UNSPECIFIED = "unspecified"


class StimulusType(str, Enum):
    """Content or form of a stimulus described by the task metadata."""

    LANGUAGE = "language"
    BIOLOGICAL_SOCIAL = "biological_social"
    OBJECTS_SCENES = "objects_scenes"
    ABSTRACT_VISUAL = "abstract_visual"
    AUDITORY_NONLINGUISTIC = "auditory_nonlinguistic"
    SYMBOLIC_NUMERICAL = "symbolic_numerical"
    SOMATOSENSORY_PHYSIOLOGICAL = "somatosensory_physiological"
    NONE = "none"
    UNSPECIFIED = "unspecified"


class StimulusDynamics(str, Enum):
    """Temporal characteristics of the stimulus presentation."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    TEMPORALLY_SEQUENCED = "temporally_sequenced"
    CONTINUOUS = "continuous"
    NONE = "none"
    UNSPECIFIED = "unspecified"


class ResponseEffector(str, Enum):
    """Output effector used by the participant to respond."""

    HAND = "hand"
    FOOT = "foot"
    TONGUE = "tongue"
    OCULAR = "ocular"
    VERBAL = "verbal"
    NONE = "no_overt_response"
    RESPIRATORY = "respiratory"
    UNSPECIFIED = "unspecified"


class ResponseLaterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NONE = "none"
    UNSPECIFIED = "unspecified"


#######################################################
###### Condition-Level Ontology Building Blocks #######
#######################################################
@dataclass(frozen=True)
class ConditionOntologyTerm:
    """
    Metadata describing an ontology term.

    Attributes:
        label: The label of the ontology term.
        description: A textual description of the term.
        examples: Example instances of the term.
        annotation_guidance: Guidance for annotating this term.
    """

    label: str
    description: str | None = None
    examples: tuple[str, ...] = ()
    annotation_guidance: str | None = None


@dataclass(frozen=True)
class ConditionOntology:
    """
    Metadata describing a condition ontology.

    Attributes:
        name: The name of the ontology.
        description: A textual description of the ontology.
        terms: A dictionary mapping term labels to ConditionOntologyTerm instances.
    """

    name: str
    description: str
    terms: dict[str, ConditionOntologyTerm]

    def get(self, label: str) -> ConditionOntologyTerm:
        return self.terms[label]

    @classmethod
    def from_labels(
        cls,
        name: str,
        description: str,
        labels: set[str],
    ) -> "ConditionOntology":
        return cls(
            name=name,
            description=description,
            terms={
                label: ConditionOntologyTerm(
                    label=label,
                    description="",
                )
                for label in sorted(labels)
            },
        )


#######################################
######### Ontology Instances ##########
#######################################

STIMULUS_MODALITY = ConditionOntology(
    name="stimulus_modality",
    description="Sensory channel through which task stimuli are presented.",
    terms={
        StimulusModality.VISUAL: ConditionOntologyTerm(
            StimulusModality.VISUAL,
            "Stimuli presented through the visual sensory channel.",
            ("images", "written text", "visual patterns"),
        ),
        StimulusModality.AUDITORY: ConditionOntologyTerm(
            StimulusModality.AUDITORY,
            "Stimuli presented through the auditory sensory channel.",
            ("spoken sentences", "tones", "environmental sounds"),
        ),
        StimulusModality.AUDIOVISUAL: ConditionOntologyTerm(
            StimulusModality.AUDIOVISUAL,
            "Stimuli presented simultaneously through visual and auditory channels.",
            ("video with synchronized audio",),
        ),
        StimulusModality.NONE: ConditionOntologyTerm(
            StimulusModality.NONE,
            "No external sensory stimulus is presented.",
            ("resting state", "internally generated imagery"),
        ),
        StimulusModality.UNSPECIFIED: ConditionOntologyTerm(
            StimulusModality.UNSPECIFIED,
            "The available task description does not establish the presentation modality.",
        ),
    },
)


STIMULUS_LATERALITY = ConditionOntology(
    name="stimulus_laterality",
    description="Spatial laterality of stimulus presentation.",
    terms={
        StimulusLaterality.LEFT: ConditionOntologyTerm(
            StimulusLaterality.LEFT,
            "Stimulus presented exclusively in the left visual or sensory field.",
        ),
        StimulusLaterality.RIGHT: ConditionOntologyTerm(
            StimulusLaterality.RIGHT,
            "Stimulus presented exclusively in the right visual or sensory field.",
        ),
        StimulusLaterality.BILATERAL: ConditionOntologyTerm(
            StimulusLaterality.BILATERAL,
            "Stimulus presented to both sides simultaneously.",
        ),
        StimulusLaterality.NONE: ConditionOntologyTerm(
            StimulusLaterality.NONE,
            "Stimulus presentation has no relevant spatial laterality.",
        ),
        StimulusLaterality.UNSPECIFIED: ConditionOntologyTerm(
            StimulusLaterality.UNSPECIFIED,
            "Stimulus laterality is not established by the available information.",
        ),
    },
)


STIMULUS_TYPE = ConditionOntology(
    name="stimulus_type",
    description="Semantic content or physical form of a task stimulus.",
    terms={
        StimulusType.LANGUAGE: ConditionOntologyTerm(
            StimulusType.LANGUAGE,
            "Spoken or written linguistic material.",
            ("sentences", "narratives", "words", "pseudowords"),
        ),
        StimulusType.BIOLOGICAL_SOCIAL: ConditionOntologyTerm(
            StimulusType.BIOLOGICAL_SOCIAL,
            "Biological entities, human actions, or social interactions.",
            ("faces", "human bodies", "biological motion", "social animations"),
        ),
        StimulusType.OBJECTS_SCENES: ConditionOntologyTerm(
            StimulusType.OBJECTS_SCENES,
            "Representations of physical objects, artifacts, or environments.",
            ("tools", "houses", "natural scenes", "everyday objects"),
        ),
        StimulusType.ABSTRACT_VISUAL: ConditionOntologyTerm(
            StimulusType.ABSTRACT_VISUAL,
            "Abstract visual forms, patterns, or graphical representations.",
            ("checkerboards", "geometric shapes", "textures", "scrambled images"),
        ),
        StimulusType.AUDITORY_NONLINGUISTIC: ConditionOntologyTerm(
            StimulusType.AUDITORY_NONLINGUISTIC,
            "Auditory material without primary linguistic content.",
            ("music", "tones", "environmental sounds", "noise"),
        ),
        StimulusType.SYMBOLIC_NUMERICAL: ConditionOntologyTerm(
            StimulusType.SYMBOLIC_NUMERICAL,
            "Numerical, mathematical, or formal symbolic material.",
            ("digits", "arithmetic expressions", "mathematical symbols"),
        ),
        StimulusType.SOMATOSENSORY_PHYSIOLOGICAL: ConditionOntologyTerm(
            StimulusType.SOMATOSENSORY_PHYSIOLOGICAL,
            "Physical sensory stimulation or physiological challenges.",
            ("tactile stimulation", "thermal stimulation", "respiratory challenge"),
        ),
        StimulusType.NONE: ConditionOntologyTerm(
            StimulusType.NONE,
            "No stimulus is present or relevant for this condition.",
        ),
        StimulusType.UNSPECIFIED: ConditionOntologyTerm(
            StimulusType.UNSPECIFIED,
            "The stimulus type is not established.",
        ),
    },
)


STIMULUS_DYNAMICS = ConditionOntology(
    name="stimulus_dynamics",
    description="Temporal characteristics of stimulus presentation.",
    terms={
        StimulusDynamics.STATIC: ConditionOntologyTerm(
            StimulusDynamics.STATIC,
            "A stimulus that remains visually or perceptually unchanged during presentation.",
            ("photograph", "checkerboard", "fixed text display"),
        ),
        StimulusDynamics.DYNAMIC: ConditionOntologyTerm(
            StimulusDynamics.DYNAMIC,
            "A stimulus that changes continuously over time or depicts motion.",
            ("video", "animated shapes", "moving objects"),
        ),
        StimulusDynamics.TEMPORALLY_SEQUENCED: ConditionOntologyTerm(
            StimulusDynamics.TEMPORALLY_SEQUENCED,
            "A succession of discrete stimuli presented in a temporal sequence.",
            ("RSVP words", "sequential images", "discrete tones"),
        ),
        StimulusDynamics.CONTINUOUS: ConditionOntologyTerm(
            StimulusDynamics.CONTINUOUS,
            "An ongoing stimulus or signal presented throughout an interval.",
            ("continuous music", "ongoing sensory stimulation"),
        ),
        StimulusDynamics.NONE: ConditionOntologyTerm(
            StimulusDynamics.NONE,
            "No temporal dynamics are present or relevant for this condition.",
        ),
        StimulusDynamics.UNSPECIFIED: ConditionOntologyTerm(
            StimulusDynamics.UNSPECIFIED,
            "The temporal characteristics of presentation are not established.",
        ),
    },
)

RESPONSE_EFFECTOR = ConditionOntology(
    name="response_effector",
    description="Physical output channel used to produce a task response.",
    terms={
        ResponseEffector.HAND: ConditionOntologyTerm(
            ResponseEffector.HAND,
            "A response produced through hand movement.",
            ("button press", "finger tapping", "hand movement"),
        ),
        ResponseEffector.FOOT: ConditionOntologyTerm(
            ResponseEffector.FOOT,
            "A response produced through foot movement.",
            ("foot tapping", "pedal press"),
        ),
        ResponseEffector.TONGUE: ConditionOntologyTerm(
            ResponseEffector.TONGUE,
            "A response produced through tongue movement.",
        ),
        ResponseEffector.OCULAR: ConditionOntologyTerm(
            ResponseEffector.OCULAR,
            "A response produced through eye movement.",
            ("saccades", "smooth pursuit", "gaze shifts"),
        ),
        ResponseEffector.VERBAL: ConditionOntologyTerm(
            ResponseEffector.VERBAL,
            "A response produced through spoken or vocal output.",
            ("spoken answer", "verbal report"),
        ),
        ResponseEffector.NONE: ConditionOntologyTerm(
            ResponseEffector.NONE,
            "No overt behavioral response is required.",
            ("passive viewing", "covert mental arithmetic"),
        ),
        ResponseEffector.RESPIRATORY: ConditionOntologyTerm(
            ResponseEffector.RESPIRATORY,
            "A response involving voluntary respiratory activity.",
            ("breath holding", "paced breathing"),
        ),
        ResponseEffector.UNSPECIFIED: ConditionOntologyTerm(
            ResponseEffector.UNSPECIFIED,
            "The response effector is not established by the available information.",
        ),
    },
)


RESPONSE_LATERALITY = ConditionOntology(
    name="response_laterality",
    description="Spatial laterality of the participant's motor response.",
    terms={
        ResponseLaterality.LEFT: ConditionOntologyTerm(
            ResponseLaterality.LEFT,
            "Response performed using the left side of the body.",
        ),
        ResponseLaterality.RIGHT: ConditionOntologyTerm(
            ResponseLaterality.RIGHT,
            "Response performed using the right side of the body.",
        ),
        ResponseLaterality.BILATERAL: ConditionOntologyTerm(
            ResponseLaterality.BILATERAL,
            "Response performed using both sides of the body.",
        ),
        ResponseLaterality.NONE: ConditionOntologyTerm(
            ResponseLaterality.NONE,
            "Response has no relevant spatial laterality.",
        ),
        ResponseLaterality.UNSPECIFIED: ConditionOntologyTerm(
            ResponseLaterality.UNSPECIFIED,
            "Response laterality is not established by the available information.",
        ),
    },
)
