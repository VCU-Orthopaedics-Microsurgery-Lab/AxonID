"""
utils/__main__.py: overview of the shared AxonID modules (python -m utils).

MODULES
    recipe          The locked model, training, augmentation and morphometrics settings.
    config          Reads config.toml (calibration, folder names, reporting options).
    calibration     µm per pixel of a segmented image, corrected for downsampling.
    model_registry  Released models (models/models.json) and your own (models/user/).
    mask_contract   BGW mask values (0 / 128 / 255) and the snapping rule for other values.
    resize          Downsampling to the 1440 x 1024 working resolution.
    image           Centre crop to the 256 px patch grid.
    patching        50% patch overlap shared by training and segmentation.
    study           Finds animals, nerves and image folders in a study.
    filesystem      File listing, natural sort, repository folders.
    metrics         Dice, IoU, precision, recall, HD95.
    logger          Console output and optional log files.
    gpu             GPU detection.
    input           Validated prompts.
    console         Rich display helpers.
    version         AxonID version and environment fingerprint.

STANDALONE TOOLS (tools/)
    python -m tools.check_mask_quality --path <masks folder>   check masks before training
    python -m tools.class_balance                              class percentages of a mask folder
    python -m utils.fetch_models                               download the released models
"""


def main() -> None:
    """Print the overview above."""
    print(__doc__)


if __name__ == "__main__":
    main()
