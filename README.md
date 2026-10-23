# AxonID

Automated axon and myelin segmentation and morphometry for brightfield peripheral nerve cross-sections.

AxonID segments every axon and its myelin sheath in 40X light-microscopy images, measures each fiber (axon and fiber diameter, myelin thickness, g-ratio, shape), and compiles nerve- and study-level results (axon counts, density, diameter distributions, estimated total axon count) into Excel workbooks.

Version 1.0.0 ships two trained models and a fixed, validated processing recipe. You supply your microscope calibration; everything that defines how a model is built and how axons are counted is locked, so results from different labs and runs are comparable.

---

## Contents

1. [What AxonID 1.0.0 supports](#what-axonid-100-supports)
2. [Installation](#installation)
3. [Calibration (required)](#calibration-required)
4. [Folder layout and file names](#folder-layout-and-file-names)
5. [Running AxonID](#running-axonid)
6. [CSA tracing in QuPath](#csa-tracing-in-qupath)
7. [Outputs](#outputs)
8. [Models](#models)
9. [Training your own model](#training-your-own-model)
10. [Fixed settings (the AxonID recipe)](#fixed-settings-the-axonid-recipe)
11. [Mask format (BGW)](#mask-format-bgw)
12. [Repository layout](#repository-layout)
13. [Tools](#tools)
14. [Troubleshooting](#troubleshooting)
15. [Collaborating and contributing data](#collaborating-and-contributing-data)
16. [Citation, lineage and license](#citation-lineage-and-license)

---

## What AxonID 1.0.0 supports

| | Supported in 1.0.0 |
|---|---|
| Magnification | 40X (segmentation and morphometrics); 4X for whole-nerve CSA only |
| Stain / imaging | Toluidine Blue, brightfield |
| Training data of the released models | Rabbit peripheral nerve, control and regenerating, 1440 x 1024 px at 0.217 µm/px |
| CSA (cross-sectional area) | QuPath tracings exported as GeoJSON |
| Hardware | Any computer for segmentation and morphometrics (CPU); NVIDIA GPU for faster segmentation and for training |

Other species, stains or magnifications may work but are not validated. You can train your own model on your images with the same recipe (see [Training your own model](#training-your-own-model)).

---

## Installation

### System requirements

| | Requirement |
|---|---|
| Operating system | Windows or Linux (macOS: CPU only, not tested) |
| Python | 3.11 |
| RAM | 8 GB minimum, 16 GB recommended (peak use measured at about 2 GB) |
| GPU | Optional. An NVIDIA GPU with a current driver makes segmentation faster and is needed for training your own model |

**Measured speed** for one 1440 × 1024 px image at 40X. These are single runs on one dataset (12 images, 4,442 axons); other computers and images will differ.

| Hardware | Segmentation | Morphometrics | Peak RAM | Peak GPU memory |
|---|---|---|---|---|
| CPU only: Intel Core i7-14700 (20 cores / 28 threads), 34 GB RAM | 21 s | 3 s | 1.7 GB (segment), 2.1 GB (morphometrics) | n/a |
| GPU: NVIDIA H100 80 GB, YOUR_CPU_MODEL_CLUSTER | YOUR_VALUE_HERE | 3 s (same CPU code)* | YOUR_VALUE_HERE | YOUR_VALUE_HERE |

Only segmentation uses the GPU; morphometrics and workbook compilation run on the CPU.


### Install

You need:

* [Miniforge](https://github.com/conda-forge/miniforge): provides conda and the **Miniforge Prompt**. Conda installs the correct Python (3.11) for you. Miniforge is free for institutional use (Anaconda's default package channel, used by Anaconda and Miniconda, requires a paid licence for large organisations).
* [Git](https://git-scm.com/).
* For an NVIDIA GPU: an up-to-date NVIDIA driver.

Open **Miniforge Prompt** (Windows) or a terminal (Linux/macOS) and type one line at a time:

```bash
git clone https://github.com/VCU-Orthopaedics-Microsurgery-Lab/AxonID.git
cd AxonID
conda create -n axonid python=3.11 -y
conda activate axonid
python install.py
```

`install.py` checks that Python 3.11 is active, installs the packages, prints the detected versions, and downloads the two released models into `models/`. It installs the CPU version of PyTorch, which works on every computer. **With an NVIDIA GPU** (much faster; needed for training your own model), run `python install.py --gpu` instead; the CPU installer tells you if it finds an NVIDIA driver.

**Each time you use AxonID:** open Miniforge Prompt, then

```bash
conda activate axonid
cd path/to/AxonID
```

Paths you paste at AxonID's prompts may include quotes (as Windows "Copy as path" adds); they are removed automatically.

<details>
<summary>Without conda (Python's built-in venv)</summary>

Install Python 3.11 from [python.org](https://www.python.org/downloads/), then in the AxonID folder:

**Windows:**
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python install.py                # add --gpu for an NVIDIA GPU
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python install.py                # add --gpu for an NVIDIA GPU
```

</details>

**Models:** the model files (`.safetensors`) are not stored in git (each is about 150 MB, over GitHub's file limit). `python -m utils.fetch_models` downloads them from the GitHub release and checks their SHA-256 and MD5 checksums. Without internet access, download them from the release page and put them in `models/`.

---

## Calibration (required)

AxonID converts pixels to micrometres with your microscope's calibration. **Set it before measuring anything.** A wrong value does not cause an error: every µm, µm² and density result is silently wrong.

For each magnification you use (40X, and 4X if you will measure whole-nerve CSA), with the same microscope, objective, camera and export settings as your nerve images:

1. **Get an image with a known length in it.** Either
   * **burn a scale bar into an image** with your acquisition software (how the AxonID lab calibrated; see below), or
   * **image a stage micrometer** (a slide with a ruled scale of known spacing).
2. **Measure that length in pixels.** Open the image in [Fiji](https://fiji.sc/), draw a straight line exactly along the scale bar (or across as many micrometer divisions as fit in the field), and read the length in pixels (Analyze > Measure; if Fiji reports µm instead of pixels, first remove the scale with Analyze > Set Scale > Click to Remove Scale). Measure the longest length available and repeat a few times: a 1-pixel error on a 230-pixel bar is 0.4% in length and 0.9% in area.
3. **Calculate µm per pixel** = known length in µm / measured length in pixels.
4. **Note the width in pixels** of the image files your camera saves (Fiji: Image > Show Info, or the file properties).
5. **Enter the value in `config.toml`** (a plain text file in the AxonID folder; open it with any text editor) under the table for that magnification, one line per image width (`<width> = <µm per pixel>`):

```toml
[pixel_size_um.4X]
1440 = 2.174
2880 = 1.087

[pixel_size_um.40X]
1440 = 0.217
```

If your camera saves the same field at two widths (for example 1440 and 2880 pixels), the µm per pixel halves when the width doubles; enter both.

**How the AxonID lab calibrated (the values shipped in `config.toml`).** Nikon Eclipse E600 with a Nikon Fi3 camera and NIS-Elements BR 6.02.03. A scale bar was burned into an image at each magnification with NIS-Elements and its length measured in pixels on the 1440-pixel-wide image:

| Magnification | Scale bar | Measured | µm/px at 1440 px wide | µm/px at 2880 px wide |
|---|---|---|---|---|
| 4X | 500 µm | 230 px | 2.174 | 1.087 |
| 40X | 50 µm | 230 px | 0.217 | |

These values are examples: they apply only to that microscope and camera.

**Burned-in scale bar or stage micrometer?** A scale bar drawn by the acquisition software is only as accurate as that software's own calibration for the objective and camera. Measuring it recovers the software's value; it does not check it. If the software calibration was itself set with a stage micrometer, the two methods agree. If you are unsure how it was set, image a stage micrometer once at each magnification and compare: a stage micrometer is the independent reference.

**Always use the original image width.** Segmentation downsamples images larger than 1440 x 1024 px before predicting (the released models were trained at that scale) and records the original size and the scale factor in each segmented TIFF. Morphometrics then corrects the pixel size automatically:

```
µm per segmented pixel = pixel_size_um[magnification][original width] / resize scale
```

QuPath tracings are made on the original images, so CSA uses the original calibration directly. Each morphometrics workbook reports the pixel size, original size, resize scale and calibration source it used (Summary sheet).

**Why 4X images are needed.** The 40X images sample only a representative part of a nerve, so the axons counted in them are not the nerve's total. AxonID estimates the whole-nerve count by scaling: it divides the axons counted in the 40X images by the tissue area traced in those images (the axon density), then multiplies by the whole-nerve cross-sectional area traced on the 4X image(s):

```
estimated total axons = axons counted at 40X / traced 40X tissue area x traced 4X whole-nerve area
```

The estimate is only as good as the sample: it assumes the 40X fields are representative of the whole cross-section, and it needs a CSA tracing for every 40X image and for the 4X image(s). Each magnification is converted to µm² with its own calibration, which is why both 40X and 4X need an entry in `pixel_size_um`.

If no calibration matches, AxonID reports measurements in pixels and warns you.

---

## Folder layout and file names

AxonID works on a study folder organised by animal and nerve:

```text
MyStudy/                                   study
└── Rb41/                                  animal
    └── Rb41_Left_E/                       nerve
        ├── 40X_TIFF/                      40X images to segment (.tif / .tiff)
        │   ├── Rb41_Left_E_40X_001.ome.tif
        │   └── Rb41_Left_E_40X_002.ome.tif
        ├── 4X_TIFF/                       whole-nerve image(s), for CSA only
        │   └── Rb41_Left_E_4X_001.ome.tif
        ├── CSA/                           QuPath GeoJSON tracings (see below)
        │   ├── Rb41_Left_E_40X_001_CSA.geojson
        │   ├── Rb41_Left_E_40X_002_CSA.geojson
        │   └── Rb41_Left_E_4X_001_CSA.geojson
        ├── Segmented/                     created by segmentation
        ├── Morphometrics/                 created by morphometrics
        └── QC/                            review images (optional)
```

Naming convention (used by the AxonID lab; the 4X rule below depends on it):

```text
nerve folder:  <AnimalID>_<Side>_<Segment>                 e.g. Rb41_Left_E
image files:   <nerve folder name>_<Mag>_<###>.tif          e.g. Rb41_Left_E_40X_001.ome.tif, Rb41_Left_E_4X_001.tif
```

Rules:

* Image folders are named `<magnification>_TIFF`, for example `40X_TIFF` and `4X_TIFF`. Upper or lower case both work (`40x_tiff`). A nerve without such a folder is skipped without an error. (The suffix is `tiff_suffixes` under `[folders]` in `config.toml`.)
* 4X images go in the nerve's `4X_TIFF` folder and must be named `<nerve folder name>_4X_<anything>`, as `.tif`, `.tiff`, `.ome.tif` or `.ome.tiff` (any upper/lower case). A nerve can have several 4X images; their CSAs are added.
* Each CSA file is named after its image without the extension: image `X.ome.tif` (or `X.tif`) gives `CSA/X_CSA.geojson`.
* `segment` and `morphometrics` accept a study, animal or nerve folder. `batch_axon` needs the study folder.
* Image format. Use 8-bit grayscale or RGB TIFFs (8-bit is recommended). 16-bit images are converted to 8-bit on loading. If your camera stores 12- or 14-bit data unscaled in a 16-bit file, AxonID rescales it and prints a warning; check the result, or export the image as 8-bit. 32-bit floating-point and multi-channel (more than 3) images are rejected with an error.

---

## Running AxonID

Run every command from the AxonID folder with the environment activated (`python -m` needs to start there to find AxonID's modules). Each command asks for the folder to process when it starts; you can give it a study, an animal or a single nerve folder (`batch_axon` needs a study folder).

There are two ways to work: **step by step** while you check the results, and **Batch_Axon** once you trust them.

### Step by step: spot-check before trusting the output

Use this for your first nerves, a new microscope or stain, or a new model. Run each stage on one nerve or one animal and look at the result before moving on.

```bash
python -m segment          # 1. segmentation: 40X images -> BGW masks in Segmented/
python -m morphometrics    # 2. per-axon measurements -> Morphometrics/
```

**1. Segmentation** lists the available models (recommended model first, then your own) and writes one BGW TIFF per image (`<image>_segmented.tif`) to the nerve's `Segmented/` folder. Each segmented TIFF records the model, recipe, original size and resize scale.
*Check:* the overlays in the nerve's `QC/` folder. Are axons (white) and myelin (grey) where they should be, including in regenerating tissue and near the edges of the tissue?

**2. Morphometrics** separates touching axons, pairs each axon with its fiber, measures every pair and writes one workbook per image plus a diameter distribution per nerve to `Morphometrics/`. If an image has a CSA tracing (see [CSA tracing in QuPath](#csa-tracing-in-qupath)), only the axons inside it are measured.
*Check:* the watershed figure in `QC/` (are touching axons split correctly, and fused ones not over-split?), the Summary sheet of a few workbooks (pixel size and calibration source as expected?), and whether axon counts and mean g-ratio are plausible for the tissue.

Both stages skip work that is already done (images with a segmentation, datasets with complete workbooks), so you can rerun them safely. To redo a nerve, delete its `Segmented/` or `Morphometrics/` folder first.

### Batch_Axon: a whole study in one run

Once the step-by-step results look right, process the whole study with one command:

```bash
python -m batch_axon
```

It asks for the study folder and the model, then runs without further input: it segments every nerve, runs morphometrics, and compiles everything, including CSA (QuPath tracings) and estimated total axon counts, into one workbook per animal and one for the study. A full study can take hours; keep the terminal open and the computer awake. Because finished images and datasets are skipped, an interrupted run continues where it stopped when you start it again.

Options:

```bash
python -m batch_axon                       # segment + morphometrics + compile (default)
python -m batch_axon --skip-segment        # segmentation already done: morphometrics + compile
python -m batch_axon --skip-morphometrics  # morphometrics already done: segment + compile
python -m batch_axon --compile-only        # only rebuild the study workbook from the animal workbooks
```

`--compile-only` keeps manual corrections you made in the animal workbooks, for example after reviewing flagged values. Set `batch_axon = true` under `[logging]` in `config.toml` to keep a log file of the run in the study folder.

---

## CSA tracing in QuPath

CSA (cross-sectional area) is needed for axon density and for the estimated total axon count. Trace it in [QuPath](https://qupath.github.io/) (tested on v0.7.0) on the **original** images.

1. Create a QuPath project per cohort and add the 40X images (and the 4X whole-nerve images) from the study folders.
2. For each image, draw the region to measure with the Polygon or Brush tool and set the annotation **Name** to `+`. Draw regions to exclude (for example epineurium or artifacts inside the traced area) and name them `-`. Several `+` and `-` regions are allowed: AxonID measures the union of `+` minus the union of `-`.
   * 40X images: trace all the nerve tissue in the field that contains axons. Axons outside the tracing are not measured.
   * 4X images: trace the whole nerve.
3. Open `batch_axon/overlay/export_csa_geojson.groovy` in QuPath (Automate > Script editor).
4. Run it once as is (`DRY_RUN = true`): it lists what it would export and every annotation that needs attention (untagged, open paths, zero area).
5. Fix any problems, set exactly `DRY_RUN = false` (lowercase `false`) and run again. The GeoJSONs are written to each nerve's `CSA/` folder.
6. Use **Run**, not "Run for project": the script already loops over every image in the project.

Lines (polylines) and unclosed polygons have no area and are not exported; re-trace them as closed polygon shapes.

**The tracing decides which axons are measured.** When an image has a tracing, morphometrics measures only the axons whose centre lies inside it (the `+` regions minus the `-` regions). Counts, g-ratios, diameters, distributions, densities and the estimated total axons then all refer to the traced tissue. The axons outside the tracing are not lost: their number is on the image's Summary sheet (*Accepted axons outside tracing*), they are listed on the image's *Outside tracing* sheet, and all of a nerve's excluded axons are pooled in `Morphometrics/<nerve>_outside_tracing.xlsx` with their position in the original image (`qupath_x`, `qupath_y`: the pixels QuPath uses, x = column, y = row), so each one can be found again. A large share outside means that the tracing leaves out axon-bearing tissue, or that something outside the nerve was segmented; check the `QC/` image.

**Trace before you run morphometrics.** An image with no tracing yet is measured in full, and morphometrics skips images that are already done, so a tracing drawn afterwards is not applied automatically. Batch_Axon warns when it finds this. To apply the tracing, move the nerve's `Morphometrics/` folder aside (do not delete it) and run morphometrics again.

---

## Outputs

### Per image: `Morphometrics/<image>_morphometrics.xlsx`

| Sheet | Contents |
|---|---|
| Summary | Model, recipe, pixel size, original size, resize scale, whether a CSA tracing was applied and how many axons fell outside it, morphometrics rules, headline means |
| Axon | Per axon: area, equivalent diameter, major/minor axis, solidity, eccentricity, orientation, position |
| Myelin+G-ratio | Per axon: myelin area and thickness, fiber area and equivalent diameter, g-ratios |
| Fiber | Per fiber: area, equivalent diameter, axes, deformation, eccentricity, orientation |
| Pixel Measurements | The same accepted axons in pixel units, for checking calibration |
| Outside tracing | Only when a CSA tracing excluded axons: those axons (position in the segmented and the original image, diameter, g-ratio). They are not in the other sheets |

Every row also carries `resolution` (segmented size), `original_resolution`, `resize_scale` and `px_size_um`.

**G-ratio** is reported three ways for every fiber:

* `gratio_equiv_diam` = axon equivalent diameter / fiber equivalent diameter
* `gratio_mean_axes` = mean axon axis length / mean fiber axis length
* `gratio` = the mean of the two

`primary_gratio_method` (under `[reporting]` in `config.toml`) selects which one every summary uses (the per-image Summary sheet, distributions, and the animal and study workbooks; default `equiv_diam`).

**Acceptance rules:** a fiber is kept when its axon equivalent diameter is at least 1.01 µm (when calibrated) and both g-ratios lie strictly between 0 and 1.

**Objects cut by the image border** are kept and measured from the part that is visible. A cut axon's diameter is underestimated; its g-ratio changes little, because the axon and its myelin are cut together. Cut objects are also counted, so counts and densities are slightly high: in YOUR_DATASET, YOUR_VALUE_HERE% of the axon objects touched the image border. AxonID 1.0.0 applies no border rule, because that would change the validated measurements. The effect is similar in images of the same size and magnification, so compare groups that were imaged the same way.

### Per nerve: `Morphometrics/<nerve>_binned.xlsx`

Pooled fiber-diameter distributions at three resolutions (0.5, 2 and 5 µm bins; set under `[morphometrics_bins]` in `config.toml`), with count, percent, cumulative percent and g-ratio mean and SD per bin.

### Per nerve: `Morphometrics/<nerve>_outside_tracing.xlsx`

Written only when a CSA tracing excluded axons. One row per excluded axon across the nerve's images: image, label, `qupath_x` and `qupath_y` (position in the original image, in QuPath's pixels), position in the segmented image, diameters and g-ratio. These axons are in no count, mean, distribution or density.

### Per animal and per study (Batch_Axon)

`<Animal>/<animal>_Data.xlsx` and `<Study>/<study>_Data.xlsx`: one block per nerve with per-image CSA (µm²), axon count, g-ratio, axon diameter and density, a totals row, the 4X whole-nerve CSA, and

```
estimated total axons = total sampled axons / total sampled 40X CSA x 4X whole-nerve CSA
```

The estimate is given only when every 40X image has a CSA and the 4X CSA is complete; otherwise the cell reads `missing` (red). Derived cells are live Excel formulas, so correcting a value after review updates them. A measured 0 is highlighted yellow for review; an image with no axons stays in the totals (with its CSA) rather than being dropped. The last two columns, *Axons Outside Tracing* and *Outside Tracing (%)*, show how many accepted axons lay outside the image's CSA tracing and were excluded from every other number in the row (the share is outside / (outside + counted)); they read `n/a` for an image without an applied tracing, and anything above 0 is highlighted yellow for review. The totals row gives the sum and the share over the images that had a tracing. In the totals row, g-ratio and axon diameter are averaged over all axons of the nerve (each image's mean weighted by its axon count), the same pooling as total axons and the distributions. The study workbook has a Provenance sheet.

### Review images: `QC/`

Segmentation overlays and watershed figures (axon and fiber separation, matched centroids), controlled by `[qc]` in `config.toml`. For images with a CSA tracing, the watershed figure also draws the tracing outline (yellow; dashed around excluded `-` regions) and marks each axon outside it with a cyan x; the row title gives how many were excluded.

---

## Models

| File | Augmentation | Default | SHA-256 |
|---|---|---|---|
| `axonid_v1_unetpp_dn169_aug.safetensors` | yes | recommended | `ebc5c858317ecfec1db5e484dc707af08ad508e1547b7ce545d3ee7432a41469` |
| `axonid_v1_unetpp_dn169.safetensors` | no | alternative | `7a614dc7c7a18008117e87c6e2232202522053e06316de2e563edf40204b5054` |

Both are UNet++ with a DenseNet169 encoder, trained with the AxonID recipe on 30 rabbit 40X images (Toluidine Blue, brightfield; 15 control, 15 regenerating): 20 for training and 10 for validation (phenotype-stratified, seed 1). They differ only in training augmentation. In the AxonID lab's evaluation on independent holdout images, augmentation gave a small average improvement in instance-level accuracy; the evaluation has not been published yet.

Model files use the [safetensors](https://github.com/huggingface/safetensors) format, which holds only weights and text: opening one cannot run code, unlike older `.pt` files. Each model file stores its training record (recipe, split sizes, best epoch, environment) and the name of the original study run (`source_run_id`). `models/models.json` lists the released models; models you train appear in the segmentation menu automatically from `models/user/`.

---

## Training your own model

Train on your own images when the released models do not fit your tissue, stain or microscope. The architecture and training settings are fixed (see [Fixed settings](#fixed-settings-the-axonid-recipe)); you provide images and masks.

### Data

```text
MyTraining/
├── images/      ctrl_<name>.tif, regen_<name>.tif, ...
└── masks/       the same file names, BGW masks (.png or .tif)
```

* Masks use the [BGW format](#mask-format-bgw): 0 background, 128 myelin, 255 axon, 8-bit grayscale.
* Name images `ctrl_...` and `regen_...` (control and regenerating tissue). The validation set (one third of the images) is then drawn separately from each group, so both are represented. Other names work, but the split is then random by image, with a warning.
* Images larger than 1440 x 1024 px are downsampled exactly as in segmentation; smaller images are used at their own size.
* Check the masks first: `python -m tools.check_mask_quality --path MyTraining/masks`. Grey values other than 0/127/128/255 are snapped to the nearest class during training (0-63 to background, 64-191 to myelin, 192-255 to axon); the tool and the training log report how many pixels that affects.

### Making masks (GIMP)

Masks can be drawn in any image editor that saves exact grey values; the AxonID lab uses [GIMP](https://www.gimp.org/) (free).

1. Open the image in GIMP. Add a new transparent layer above it for tracing.
2. Trace every myelin sheath in grey **128** (`#808080`), then every axon on top in white **255** (`#FFFFFF`). Use the **Pencil** tool and the **Bucket Fill** tool with **Antialiasing off** and **Feather edges off**. Smoothed (anti-aliased) edges create in-between grey values.
3. Leave everything else empty; it becomes background (0, black) in step 4.
4. Hide or delete the image layer, add a black background layer underneath, and merge the layers into one (Image > Flatten Image).
5. Convert to grayscale (Image > Mode > Grayscale) and export as PNG with exactly the same name as the image (File > Export As, `<image name>.png`).
6. Run `python -m tools.check_mask_quality` on the masks folder: it reports any values other than 0/127/128/255 and any mask whose size differs from its image.

**Start from AxonID's segmentation instead of a blank layer.** For images up to 1440 x 1024 px, the segmented TIFF from `python -m segment` is already a BGW mask of the same size: open it in GIMP, correct its mistakes with the same colours and settings, and export it as above. Correcting is usually much faster than tracing from scratch. Mark every axon in the image, including small and regenerating ones: an untraced axon teaches the model that it is background.

Practical sizes: 10 images (5 + 5) minimum, 20 good, 30 matches the released models. Images from different animals help more than many images from one animal; regenerating images add the most variety.

### Run

```bash
python -m train                              # interactive
python -m train --config train_config.toml   # non-interactive (cluster jobs)
```

`train_config.toml` (only `images_dir` is required; delete a line to use the default):

```toml
images_dir   = 'C:\Data\MyTraining'   # single quotes keep Windows paths as typed
model_name   = ""
augmentation = false
epochs       = 400
batch_size   = 128
seed         = 1
```

* **augmentation**: the AxonID augmentation recipe (flips, rotation, elastic deformation, gamma, noise, blur), applied once to each training patch before training.
* **epochs**: upper limit; early stopping normally ends training much sooner.
* **batch_size**: 128 needs a large GPU (the released models were trained on 80 GB). On a smaller GPU use a smaller batch; AxonID tells you when the GPU runs out of memory.
* **seed**: controls the split, the augmentation and the training order. Seed 1 on the released models' 30 images reproduces their validation set.

Changing `epochs` or `batch_size` is allowed but recorded in the checkpoint as a recipe deviation. Any other key is refused.

### Output

* `models/user/<model_name>.safetensors`: the best-validation-loss weights and the full training record. Existing models are never overwritten.
* `logs/<model_name>_training_log.txt`: the run, epoch by epoch, with final Dice, IoU, precision, recall and HD95 on the validation set (per phenotype when names are prefixed).
* Patches are saved in `images/cropped/` and `masks/cropped/` and reused by later runs. Delete both folders after changing the images or masks.

---

## Fixed settings (the AxonID recipe)

These settings define a validated AxonID model and measurement. They live in `utils/recipe.py`. If you attempt to override them by adding them to `config.toml`, AxonID will halt and display an error explaining that these settings are locked.

| Area | Setting |
|---|---|
| Network | UNet++, DenseNet169 encoder (ImageNet initialisation), 1 input channel, 3 classes |
| Loss | 0.5 weighted Dice + 0.5 weighted cross-entropy; class weights background 1, myelin 2, axon 2 |
| Optimiser | AdamW, learning rate 1e-3, weight decay 0.01 |
| Schedule | ReduceLROnPlateau (patience 15, factor 0.5, minimum 1e-6) |
| Early stopping | validation loss, patience 40, minimum change 0.001; at most 400 epochs |
| Patches | 256 x 256 px, 50% overlap; images downsampled to fit 1440 x 1024 px |
| Precision | full (32-bit float); mixed precision is not used |
| Split | one third of images for validation, per phenotype |
| Augmentation | 50% of patches unchanged; otherwise h-flip 5%, v-flip 20%, rotation ±45° 10%, elastic 5%, gamma 0.8-1.2 10%, noise 5%, blur 5% |
| Watershed | distance threshold 0.08, background disk 5 px, minimum seed 2 px |
| Acceptance | axon equivalent diameter ≥ 1.01 µm; both g-ratios in (0, 1) |

The watershed and acceptance rules were validated on the ground-truth masks of the AxonID holdout images: 7746 axons counted against 7730 annotated (0.2%).

Input normalisation differs on purpose between training (per patch column) and segmentation (per patch row); segmentation's convention was kept after a direct comparison. See the note in `utils/recipe.py`.

What you can change in `config.toml`: calibration, the headline g-ratio method, distribution bins, review images, logging and folder names.

---

## Mask format (BGW)

Segmentations and training masks share one 8-bit grayscale format:

| Value | Class |
|---:|---|
|   0 | background |
| 128 | myelin (127 is also read as myelin) |
| 255 | axon |

Other grey values in training masks are snapped to the nearest class. Masks saved as RGB, or with smoothing (anti-aliasing) at the edges, are the usual cause of off-format values.

---

## Repository layout

```text
AxonID/
├── segment/          python -m segment: model menu, patch inference with Hann blending, BGW output
├── morphometrics/    python -m morphometrics: watershed, axon/fiber matching, measurements, distributions
├── batch_axon/       python -m batch_axon: study workbooks; overlay/ holds the QuPath export script
├── train/            python -m train: preprocessing, image-level split, augmentation, training loop
├── utils/            shared modules; recipe.py holds every fixed setting
├── tools/            mask checks, class balance, .pt to .safetensors conversion
├── tests/            automated tests (python -m pytest)
├── models/           models.json (released models), user/ (models you train)
├── config.toml       your settings (calibration, reporting, folder names)
├── train_config.toml template for non-interactive training
├── install.py        installs packages and models (python install.py [--gpu])
└── requirements.txt
```

---

## Tools

```bash
python -m tools.check_mask_quality --path <masks folder>   # check masks before training
python -m tools.class_balance                              # class percentages of a mask folder
python -m utils.fetch_models                               # download / re-check the released models
python -m tools.convert_model <model.pt>                   # convert a trusted older .pt model to .safetensors
python utils/version.py                                    # versions and GPU
python -m utils                                            # overview of the shared modules
```

**Automated tests** (synthetic data only, about 10 seconds, no models needed):

```bash
python -m pip install pytest
python -m pytest
```

They cover mask snapping, calibration, the train/validation split, augmentation, CSA and 4X discovery, settings validation, patch geometry, and watershed counts and g-ratios on synthetic fibers of known size. Run them after changing any code.

---

## Troubleshooting

| Problem | Cause and fix |
|---------|---------------|
| "config.toml contains settings that cannot be changed" | Recipe settings (training, watershed, quality_filters, clahe, ...) were added to `config.toml`. Remove the listed entries. |
| "config.toml could not be read" | A typo in `config.toml`; the message gives the line. Usually a missing quote or bracket. Text needs quotes, numbers and true/false do not. |
| "config.toml not found" but you have a `config.json` | `config.json` is from an earlier version. Copy your `pixel_size_um` values into `config.toml`. |
| A nerve is not found | Its image folder name does not end in one of `tiff_suffixes` (e.g. `40X_TIFF`). |
| Measurements in pixels instead of µm | No `pixel_size_um` entry for that magnification and original image width. |
| "no resize record" warning | The images were segmented by an older version. Pixel size is looked up at the segmented width, assuming no downsampling. Re-segment to remove the warning. |
| Warning that a CSA tracing "did not apply" | The tracing was drawn after morphometrics ran, so axons outside it are still counted. Move the nerve's `Morphometrics/` folder aside (do not delete it) and run morphometrics again. |
| CSA "missing" | No `CSA/<image>_CSA.geojson`, no `+` annotation in it, or the file name does not match the image. Run the QuPath script with `DRY_RUN = true` to see what is missing. |
| "checksum does not match" | The model file is incomplete or not the released file. Delete it and run `python -m utils.fetch_models`. |
| "AxonID loads .safetensors model files only" | The model is an older `.pt` file. If you made it or trust its source, convert it with `python -m tools.convert_model <file.pt>` and put the `.safetensors` file in `models/user/`. |
| GPU out of memory during training | Use a smaller `batch_size`. |
| "AxonID needs Python 3.11" when installing | The axonid environment is not active, or was created with another Python. Run `conda activate axonid`; if it persists, recreate it with `conda create -n axonid python=3.11 -y`. |
| `git clone` fails with an SSL / certificate error (common on hospital and university networks) | Run `git config --global http.sslBackend schannel` (Windows) so Git uses the certificates Windows already trusts, then clone again. Do not turn certificate checking off (`http.sslverify false`). |
| Training: "patch folders are out of date" | Images or masks changed after patches were made. Delete `images/cropped` and `masks/cropped`. |

---

## Collaborating and contributing data

AxonID's models improve with annotated images from more species, stains, magnifications and microscopes. If your lab has, or plans to make, BGW masks for its own nerve images, we would like to hear from you, whether to train a model for your tissue together or to contribute images to future AxonID releases.

Please contact us by opening an issue on this repository or by email marc.mazur@vcuhealth.org with:

* species, nerve, stain and magnification;
* microscope, camera and calibration (µm per pixel);
* how many images are annotated (control and regenerating, number of animals) and how the masks were made.

Contributed data stays yours: how it is used, shared and credited (acknowledgement or co-authorship) is agreed with each lab before any training. Images from human tissue require the appropriate ethics approval and data-use agreement at both institutions.

---

## Citation, lineage and license

**Citation:** No publication describing AxonID is available yet. Until one is, please cite the software: AxonID 1.0.0, VCU Orthopaedics-Microsurgery Lab (2026), https://github.com/VCU-Orthopaedics-Microsurgery-Lab/AxonID.

**Lineage:** AxonID is derived from **DeepAxon** by Kush Savsani (2024), preserved in [kushsavsani/deepaxon-legacy](https://github.com/kushsavsani/deepaxon-legacy). AxonID 1.0.0 (2026) is developed by the VCU Orthopaedics-Microsurgery Lab (Marc Mazur) and replaces the earlier code with a PyTorch pipeline, the locked recipe, morphometrics and study compilation.

**License:** MIT. See [LICENSE.md](LICENSE.md) for the copyright and attribution notices.
