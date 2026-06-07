"""
Shared configuration for the static probe replication of Chen et al.,
plus the dynamic-attribute extension (epistemic state, emotion).
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CONV_DIR = DATA_DIR / "conversations"   # raw jsonl files
ACT_DIR = DATA_DIR / "activations"     # extracted hidden states
PROBE_DIR = DATA_DIR / "probes"          # trained probe checkpoints
RESULT_DIR = DATA_DIR / "results"        # accuracy jsons + figures
COT_DIR = DATA_DIR / "cot_responses"   # chain-of-thought intervention outputs

for _d in (CONV_DIR, ACT_DIR, PROBE_DIR, RESULT_DIR, COT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
# Switch between models here. Mistral-7B requires no HuggingFace access request.
# LLaMA-2-13B matches the paper exactly but requires Meta approval at:
#   https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
#
# MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"; NUM_LAYERS = 32; HIDDEN_DIM = 4096
MODEL_ID   = "meta-llama/Llama-2-13b-chat-hf"
NUM_LAYERS = 40       # 40 transformer blocks; index 0 = embed, 1-40 = blocks
HIDDEN_DIM = 5120     # LLaMA-2-13B residual stream width

# ── Attributes ────────────────────────────────────────────────────────────────
# Each value has:
#   subcategories : short keys used in filenames / one-vs-rest labels
#   labels        : human-readable names matching the paper
#   n_convos      : target generation count (Table 1)
#   prompt_desc   : description injected into the GPT generation prompt
#   probe_suffix  : text appended after the last user turn for activation extraction

ATTRIBUTES = {
    "age": {
        "subcategories": ["child", "adolescent", "adult", "older_adult"],
        "labels": ["Child (< 13)", "Adolescent (13-17)", "Adult (18-64)", "Older Adult (> 64)"],
        "n_convos": 4000,   # ~1000 per subcategory
        "prompt_desc": {
            "child":        "a child user who is younger than 13 years old",
            "adolescent":   "an adolescent user who is between 13 and 17 years old",
            "adult":        "an adult user who is between 18 and 64 years old",
            "older_adult":  "an older adult user who is over 64 years old",
        },
        "probe_suffix": "I think the age of this user is",
    },
    "gender": {
        "subcategories": ["male", "female"],
        "labels": ["Male", "Female"],
        "n_convos": 2400,   # ~1200 per subcategory
        "prompt_desc": {
            "male":   "a male user",
            "female": "a female user",
        },
        "probe_suffix": "I think the gender of this user is",
    },
    "education": {
        "subcategories": ["some_schooling", "high_school", "college_and_beyond"],
        "labels": ["Some Schooling", "High School", "College & Beyond"],
        "n_convos": 4500,   # ~1500 per subcategory
        "prompt_desc": {
            "some_schooling":      "a user with some schooling (less than a high school diploma)",
            "high_school":         "a user whose highest education is a high school diploma",
            "college_and_beyond":  "a user with a college degree or higher",
        },
        "probe_suffix": "I think the education level of this user is",
    },
    "socioeconomic": {
        "subcategories": ["lower", "middle", "upper"],
        "labels": ["Lower", "Middle", "Upper"],
        "n_convos": 3000,   # ~1000 per subcategory
        "prompt_desc": {
            "lower":  "a user from a lower socioeconomic background",
            "middle": "a user from a middle socioeconomic background",
            "upper":  "a user from an upper socioeconomic background",
        },
        "probe_suffix": "I think the socioeconomic status of this user is",
    },
}

# ── Probe training ─────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.8    # 80 / 20 train-val split (paper §4.2)
L2_WEIGHT_DECAY = 1e-4  # L2 regularisation
PROBE_LR = 1e-3
PROBE_EPOCHS = 30
PROBE_BATCH = 128
SEED = 42

# ── Generation ────────────────────────────────────────────────────────────────
GEN_MODEL = "gpt-4o-mini"   # swap to "gpt-4o-mini" for cheaper inference
N_TURNS = 7                 # average turns per conversation (paper: 7.5)
CONVOS_PER_CALL = 1                 # one conversation per API call

# ── Dynamic attributes — organised by taxonomy ────────────────────────────────
#
# Taxonomy source: user-state modelling literature
#
#   ┌────────────────────────────────────────────────────────┐
#   │ 1. Affective State   (Emotion, Mood, Arousal)          │
#   │ 2. Cognitive State   (Proficiency, Intent, Attention)  │
#   │ 3. Interaction Style (Pacing, Formality, Sentiment)    │
#   │ 4. Contextual State  (Temporal, Device, Task) — skip   │
#   └────────────────────────────────────────────────────────┘
#
# Why skip Contextual State? Temporal location, device, and task type are
# not reliably inferable from text alone, so probes would capture surface
# lexical cues rather than a genuine internal user model.
#
# Safety argument: attributes in categories 1–3 are exactly the signals an
# adversarial model would track to adapt manipulation strategies in real-time.
#
# Each attribute carries:
#   taxonomy_category  : string label from the taxonomy above
#   transition_pairs   : (sub_A → sub_B) pairs for state-change conversations
#   cot_questions_file : flat .txt file, one question per line
#   cot_metrics        : metrics from measure_cot_complexity() most sensitive
#                        to this attribute (guides which plots to prioritise)

DYNAMIC_ATTRIBUTES = {

    # ── 1. Affective State ────────────────────────────────────────────────────
    # Hypothesis: a model tracking emotional valence will produce more
    # reassuring / apologetic CoT when frustration is patched in.

    "emotional_state": {
        "taxonomy_category": "affective",
        "subcategories": ["frustrated", "neutral", "enthusiastic"],
        "labels": ["Frustrated", "Neutral", "Enthusiastic"],
        "n_convos": 2400,
        "prompt_desc": {
            "frustrated":   "a frustrated user who is impatient, uses short terse messages, "
                            "and has expressed dissatisfaction with prior answers",
            "neutral":      "a user with a neutral emotional tone",
            "enthusiastic": "an enthusiastic user who shows excitement, uses exclamation "
                            "marks, and is eager to learn more",
        },
        "probe_suffix": "I think the emotional state of this user is",
        "transition_pairs": [("enthusiastic", "frustrated"), ("frustrated", "enthusiastic")],
        "cot_questions_file": "data/causality_test_questions/emotional_state.txt",
        "cot_metrics": ["hedging_frequency", "reassurance_count", "response_length",
                        "apology_count", "flesch_kincaid_grade"],
    },

    # ── 2. Cognitive State ────────────────────────────────────────────────────
    # Proficiency — does the model track how much the user knows?
    # Hypothesis: patching novice → expert shifts CoT from analogies to
    # formal definitions, reducing FK grade and increasing technical density.

    "knowledge_level": {
        "taxonomy_category": "cognitive_proficiency",
        "subcategories": ["novice", "intermediate", "expert"],
        "labels": ["Novice", "Intermediate", "Expert"],
        "n_convos": 3000,
        "prompt_desc": {
            "novice":        "a novice user with little background knowledge on the topic, "
                             "who asks basic clarifying questions and needs things explained simply",
            "intermediate":  "a user with moderate background knowledge who is comfortable "
                             "with standard terminology but needs some concepts clarified",
            "expert":        "an expert user who is highly knowledgeable, uses precise "
                             "technical language, and expects detailed technical responses",
        },
        "probe_suffix": "I think the knowledge level of this user is",
        "transition_pairs": [("expert", "novice"), ("novice", "expert")],
        "cot_questions_file": "data/causality_test_questions/knowledge_level.txt",
        "cot_metrics": ["explanation_depth", "technical_density", "hedging_frequency",
                        "analogy_count", "flesch_kincaid_grade"],
    },

    # Attention/comprehension — does the model track whether the user is
    # following?  Hypothesis: patching confused → clear reduces repetition
    # and increases the model's willingness to advance the topic.

    "confusion": {
        "taxonomy_category": "cognitive_attention",
        "subcategories": ["confused", "neutral", "clear"],
        "labels": ["Confused", "Neutral", "Clear"],
        "n_convos": 2400,
        "prompt_desc": {
            "confused":  "a confused user who misunderstands concepts, asks contradictory "
                         "questions, and signals they are lost",
            "neutral":   "a user who seems neither confused nor particularly confident",
            "clear":     "a user who is following the conversation clearly, builds on "
                         "prior answers, and asks precise follow-up questions",
        },
        "probe_suffix": "I think this user is currently",
        "transition_pairs": [("clear", "confused"), ("confused", "clear")],
        "cot_questions_file": "data/causality_test_questions/confusion.txt",
        "cot_metrics": ["explanation_depth", "hedging_frequency", "repetition_rate",
                        "example_count", "flesch_kincaid_grade"],
    },

    # Intent — what is the user trying to accomplish?
    # Hypothesis: patching "learn" → "accomplish" shifts CoT from explanatory
    # prose to step-by-step procedural output, reducing explanation_depth and
    # increasing example_count / imperative verb density.

    "user_intent": {
        "taxonomy_category": "cognitive_intent",
        "subcategories": ["learn", "accomplish", "vent"],
        "labels": ["Learn", "Accomplish", "Vent"],
        "n_convos": 2400,
        "prompt_desc": {
            "learn":      "a curious user whose primary goal is to understand a concept "
                          "deeply — they ask 'why' and 'how does this work' questions",
            "accomplish":  "a task-focused user who wants a concrete answer or step-by-step "
                           "instructions to get something done as quickly as possible",
            "vent":        "a user who is primarily expressing frustration or seeking "
                           "emotional validation rather than information",
        },
        "probe_suffix": "I think the goal of this user is to",
        "transition_pairs": [("learn", "accomplish"), ("accomplish", "learn"),
                             ("vent", "learn")],
        "cot_questions_file": "data/causality_test_questions/user_intent.txt",
        "cot_metrics": ["explanation_depth", "example_count", "response_length",
                        "hedging_frequency", "flesch_kincaid_grade"],
    },

    # ── 3. Interaction Style ──────────────────────────────────────────────────
    # Formality — the most surface-visible signal in text.
    # This is a critical test: if the model's internal formality representation
    # is causal, patching formal → casual should shift vocabulary register in
    # CoT (contractions, colloquialisms, shorter sentences) even when the
    # question itself is neutral.  This is the clearest possible demonstration
    # that the model actively uses its user profile.

    "formality": {
        "taxonomy_category": "interaction_style",
        "subcategories": ["casual", "neutral", "formal"],
        "labels": ["Casual", "Neutral", "Formal"],
        "n_convos": 2400,
        "prompt_desc": {
            "casual":  "a user who writes in a very casual, informal style — uses "
                       "contractions, slang, abbreviations (like 'gonna', 'tbh', 'lol'), "
                       "and short messages",
            "neutral": "a user with a neutral, everyday writing style",
            "formal":  "a user who writes in a formal register — full sentences, "
                       "no contractions, precise vocabulary, professional tone",
        },
        "probe_suffix": "I think the communication style of this user is",
        "transition_pairs": [("formal", "casual"), ("casual", "formal")],
        "cot_questions_file": "data/causality_test_questions/formality.txt",
        "cot_metrics": ["technical_density", "flesch_kincaid_grade", "response_length",
                        "hedging_frequency", "contraction_rate"],
    },
}

# Taxonomy category → list of attribute names (for grouped plotting)
DYNAMIC_TAXONOMY: dict[str, list[str]] = {
    "affective":             ["emotional_state"],
    "cognitive_proficiency": ["knowledge_level"],
    "cognitive_attention":   ["confusion"],
    "cognitive_intent":      ["user_intent"],
    "interaction_style":     ["formality"],
}

# Combined dict for convenience
ALL_ATTRIBUTES = {**ATTRIBUTES, **DYNAMIC_ATTRIBUTES}
