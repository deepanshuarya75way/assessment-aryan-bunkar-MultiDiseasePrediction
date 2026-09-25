import pandas as pd
from pathlib import Path


class GuidanceEngine:

    def __init__(self, database_path=None):

        # Automatically find the database folder
        if database_path is None:
            database_path = (
                Path(__file__).resolve().parent.parent / "database"
            )
        else:
            database_path = Path(database_path)

        self.database_path = database_path

        # Load database files
        self.diseases = pd.read_csv(
            database_path / "diseases.csv"
        )

        self.guidance = pd.read_csv(
            database_path / "guidance.csv"
        )

        self.probability_rules = pd.read_csv(
            database_path / "probability_rules.csv"
        )

        self.input_rules = pd.read_csv(
            database_path / "input_rules.csv"
        )

        self.sources = pd.read_csv(
            database_path / "sources.csv"
        )

    # ==================================================
    # Get disease information
    # ==================================================

    def get_disease(self, disease_id):

        disease = self.diseases[
            self.diseases["disease_id"] == disease_id
        ]

        if disease.empty:
            raise ValueError(
                f"Unknown disease_id: {disease_id}"
            )

        return disease.iloc[0]

    # ==================================================
    # Get probability level
    # ==================================================

    def get_probability_level(
        self,
        disease_id,
        probability
    ):

        rules = self.probability_rules[
            self.probability_rules["disease_id"] == disease_id
        ]

        for _, rule in rules.iterrows():

            min_probability = float(
                rule["min_probability"]
            )

            max_probability = float(
                rule["max_probability"]
            )

            if min_probability <= probability < max_probability:

                return {
                    "display_level":
                        rule["display_level"],

                    "action":
                        rule["action"]
                }

        return {
            "display_level": "general",
            "action":
                "Show general educational guidance."
        }

    # ==================================================
    # Get input-specific guidance
    # ==================================================

    def get_input_guidance(
        self,
        disease_id,
        inputs
    ):

        results = []

        if inputs is None:
            return results

        rules = self.input_rules[
            self.input_rules["disease_id"] == disease_id
        ]

        for _, rule in rules.iterrows():

            feature = rule["feature"]

            # Check whether the feature exists
            if feature not in inputs.columns:
                continue

            value = inputs.iloc[0][feature]

            # Ignore missing values
            if pd.isna(value):
                continue

            operator = rule["operator"]

            threshold = float(
                rule["threshold"]
            )

            triggered = False

            if operator == ">=":
                triggered = value >= threshold

            elif operator == ">":
                triggered = value > threshold

            elif operator == "<=":
                triggered = value <= threshold

            elif operator == "<":
                triggered = value < threshold

            elif operator == "==":
                triggered = value == threshold

            if triggered:

                results.append({
                    "feature": feature,
                    "value": value,
                    "message": rule["message"]
                })

        return results

    # ==================================================
    # Main guidance function
    # ==================================================

    def get_guidance(
        self,
        disease_id,
        probability,
        inputs=None
    ):

        # ----------------------------------------------
        # Disease information
        # ----------------------------------------------

        disease = self.get_disease(
            disease_id
        )

        disease_name = disease[
            "disease_name"
        ]

        # ----------------------------------------------
        # Probability information
        # ----------------------------------------------

        probability_info = (
            self.get_probability_level(
                disease_id,
                probability
            )
        )

        # ----------------------------------------------
        # Disease-specific guidance
        # ----------------------------------------------

        disease_guidance = self.guidance[
            self.guidance["disease_id"] == disease_id
        ]

        # ----------------------------------------------
        # Create result
        # ----------------------------------------------

        result = {

            "disease_id":
                disease_id,

            "disease_name":
                disease_name,

            "probability":
                probability,

            "probability_level":
                probability_info[
                    "display_level"
                ],

            "probability_action":
                probability_info[
                    "action"
                ],

            "consume": [],

            "limit": [],

            "lifestyle": [],

            "monitor": [],

            "seek_care": [],

            "input_flags": [],

            "sources": []
        }

        # ----------------------------------------------
        # Organize recommendations
        # ----------------------------------------------

        for _, row in disease_guidance.iterrows():

            category = row["category"]

            recommendation = (
                row["recommendation"]
            )

            if category in result:

                result[category].append(
                    recommendation
                )

        # ----------------------------------------------
        # Input-specific guidance
        # ----------------------------------------------

        result["input_flags"] = (
            self.get_input_guidance(
                disease_id,
                inputs
            )
        )

        # ----------------------------------------------
        # Sources
        # ----------------------------------------------
        #
        # Your sources.csv contains:
        #
        # source_id
        # organization
        # title
        # url
        #
        # It does not contain disease_id.
        #
        # Therefore we return the source library here.
        # ----------------------------------------------

        disease_sources = self.sources[
            (self.sources["disease_id"] == disease_id) |
            (self.sources["disease_id"] == 0)
        ]

        result["sources"] = disease_sources[
            [
                "source_id",
                "organization",
                "title",
                "url"
            ]
        ].to_dict(
            "records"
        )
        return result