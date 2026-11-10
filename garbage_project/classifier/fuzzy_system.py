import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from itertools import product


# -------------------------------
# 1. Define fuzzy input variables
# -------------------------------

model_confidence = ctrl.Antecedent(
    np.arange(0, 101, 1), "model_confidence"
)

similarity_strength = ctrl.Antecedent(
    np.arange(0, 101, 1), "similarity_strength"
)

neighbour_agreement = ctrl.Antecedent(
    np.arange(0, 101, 1), "neighbour_agreement"
)

prediction_agreement = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01), "prediction_agreement"
)


# -------------------------------
# 2. Define fuzzy output variable
# -------------------------------

similarity_weight = ctrl.Consequent(
    np.arange(0, 101, 1), "similarity_weight"
)


# -------------------------------
# 3. Membership functions
# -------------------------------

# Model confidence
model_confidence["low"] = fuzz.trapmf(
    model_confidence.universe, [0, 0, 35, 55]
)
model_confidence["medium"] = fuzz.trimf(
    model_confidence.universe, [35, 60, 80]
)
model_confidence["high"] = fuzz.trapmf(
    model_confidence.universe, [65, 85, 100, 100]
)

# Similarity strength
similarity_strength["low"] = fuzz.trapmf(
    similarity_strength.universe, [0, 0, 25, 45]
)
similarity_strength["medium"] = fuzz.trimf(
    similarity_strength.universe, [25, 50, 75]
)
similarity_strength["high"] = fuzz.trapmf(
    similarity_strength.universe, [60, 80, 100, 100]
)

# Neighbour agreement
neighbour_agreement["low"] = fuzz.trapmf(
    neighbour_agreement.universe, [0, 0, 35, 55]
)
neighbour_agreement["medium"] = fuzz.trimf(
    neighbour_agreement.universe, [35, 60, 80]
)
neighbour_agreement["high"] = fuzz.trapmf(
    neighbour_agreement.universe, [65, 85, 100, 100]
)

# Prediction agreement: 0 = disagree, 1 = agree
prediction_agreement["disagree"] = fuzz.trapmf(
    prediction_agreement.universe, [0, 0, 0.2, 0.5]
)
prediction_agreement["agree"] = fuzz.trapmf(
    prediction_agreement.universe, [0.5, 0.8, 1, 1]
)

# Output: similarity influence
similarity_weight["low"] = fuzz.trapmf(
    similarity_weight.universe, [0, 0, 20, 40]
)
similarity_weight["medium"] = fuzz.trimf(
    similarity_weight.universe, [25, 50, 75]
)
similarity_weight["high"] = fuzz.trapmf(
    similarity_weight.universe, [60, 80, 100, 100]
)


# -------------------------------
# 4. Define fuzzy rules
# -------------------------------

levels = ["low", "medium", "high"]
agreement_levels = ["disagree", "agree"]

rules = []

for model_level, sim_level, neighbour_level, agreement_level in product(
    levels, levels, levels, agreement_levels
):

    # Weak similarity evidence: don't trust similarity matching
    if sim_level == "low" or neighbour_level == "low":
        output = "low"

    # Strong similarity evidence
    elif sim_level == "high" and neighbour_level == "high":
            if model_level == "low" and agreement_level == "disagree":
                output = "high"
            else:
                output = "medium"

    # Moderate similarity evidence
    elif model_level == "low":
        output = "medium"

    elif model_level == "medium" and (
        sim_level == "high" or neighbour_level == "high"
    ):
        output = "medium"

    else:
        output = "low"

    rule = ctrl.Rule(
        model_confidence[model_level]
        & similarity_strength[sim_level]
        & neighbour_agreement[neighbour_level]
        & prediction_agreement[agreement_level],
        similarity_weight[output]
    )

    rules.append(rule)


# -------------------------------
# 5. Create fuzzy control system
# -------------------------------

fuzzy_control = ctrl.ControlSystem(rules)


# -------------------------------
# 6. Function to calculate weight
# -------------------------------

def get_similarity_weight(
    model_conf,
    best_similarity,
    neighbour_score,
    predictions_agree
):

    simulation = ctrl.ControlSystemSimulation(fuzzy_control)

    # Model confidence is already a percentage (0-100)
    simulation.input["model_confidence"] = np.clip(
        model_conf, 0, 100
    )

    # best_similarity is the raw cosine similarity (-1 to 1)
    # Convert it into a 0-100 strength value.
    similarity_percent = max(0, min(1, best_similarity)) * 100

    simulation.input["similarity_strength"] = similarity_percent

    # Neighbour score is a percentage (0-100)
    simulation.input["neighbour_agreement"] = np.clip(
        neighbour_score, 0, 100
    )

    simulation.input["prediction_agreement"] = (
        1 if predictions_agree else 0
    )

    simulation.compute()

    return round(simulation.output["similarity_weight"], 2)