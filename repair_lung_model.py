import zipfile
import json
import shutil
import os
import tempfile

source_model = "models/lung_disease_densenet121.keras"
repaired_model = "models/lung_disease_densenet121_repaired.keras"


print("Reading original model...")

with zipfile.ZipFile(source_model, "r") as zin:

    config = json.loads(zin.read("config.json"))

    def remove_incompatible_config(obj):

        if isinstance(obj, dict):

            # Remove incompatible initializer arguments
            obj.pop("input_axes", None)
            obj.pop("output_axes", None)

            for value in obj.values():
                remove_incompatible_config(value)

        elif isinstance(obj, list):

            for item in obj:
                remove_incompatible_config(item)

    remove_incompatible_config(config)

    print("Saving repaired model...")

    with zipfile.ZipFile(
        repaired_model,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as zout:

        for item in zin.infolist():

            if item.filename == "config.json":

                new_config = json.dumps(
                    config,
                    separators=(",", ":")
                ).encode("utf-8")

                zout.writestr(item, new_config)

            else:

                zout.writestr(
                    item,
                    zin.read(item.filename)
                )


print("\nRepaired model created:")
print(repaired_model)