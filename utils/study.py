"""
utils/study.py: find animals, nerves and image folders in a study.

Usage
    from utils.study import resolve_scan, detect_study_mag
    study, study_dir = resolve_scan(folder)   # study, animal, nerve or *_TIFF folder

Inputs
    Study/Animal/Nerve/<mag>_TIFF folders. Image folders are recognised by
    the config.toml tiff_suffixes (e.g. 40X_TIFF); the magnification is the
    part before the suffix. 4X folders are skipped (they hold whole-nerve
    images for CSA only).

Outputs
    {animal: {nerve: {"mag", "tiff_dir", "segmented_dir", "morphometrics_dir",
    "csa_dir"}}} and the study folder. Also StageResult, the timing record
    segment.run() and morphometrics.run() return to batch_axon.

Depends on
    utils/config.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
from dataclasses import dataclass
from pathlib import Path

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.config import load_config


@dataclass
class StageResult:
    """
    Return type for segment.run() and morphometrics.run(). Lets a caller
    (batch_axon) read a stage's own precisely-measured elapsed time and
    image count instead of independently wall-clock-timing the call from
    outside — the two numbers were previously computed twice, in two
    places, for the same span.
    
    The three optional sub-timers below are morphometrics-specific —
    segment.run() leaves them at their 0.0 default. qc_seconds is
    reported separately and deliberately excluded from any per-image
    average: QC generation is config-toggleable
    (qc.watershed/qc.segment) and will almost always be left on for
    review, so folding its cost into "time per image" would make the
    core pipeline look slower than it actually is whenever QC is on —
    which is most of the time.
    """
    success: bool
    elapsed_seconds: float
    images_processed: int
    segmentation_seconds: float = 0.0
    morphometrics_seconds: float = 0.0
    distributions_seconds: float = 0.0
    qc_seconds: float = 0.0


# ─── helpers ──────────────────────────────────────────────────────────────────
def _strip_suffix(name: str, suffix: str) -> str:
    """Strip suffix from the end of name only — not every occurrence,
    unlike str.replace()."""
    return name[:-len(suffix)] if name.endswith(suffix) else name


# ─── Study / folder scanning ──────────────────────────────────────────────────
def _has_images(d: Path, source_dir: Path | None = None) -> bool:
    """
    Return True if directory d contains image files.
    If source_dir is provided, checks that all source image stems have
    a corresponding segmented file in d.
    """
    if not d.exists():
        return False
    imgs = [f for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in ('.tif', '.tiff', '.png')]
    if not imgs:
        return False
    if source_dir is not None and source_dir.exists():
        config          = load_config()
        seg_suffix      = config.get("segmented_suffix", "_segmented")
        source_stems    = {f.stem for f in source_dir.iterdir()
                           if f.is_file() and f.suffix.lower() in ('.tif', '.tiff')}
        segmented_stems = {_strip_suffix(f.stem, seg_suffix) for f in imgs}
        return source_stems.issubset(segmented_stems)
    return True


def scan_study(study_dir: str, ignore_4x: bool = True) -> dict:
    """
    Walk a study directory and return a dict:
    {
        animal_name: {
            nerve_name: {
                'mag': '40X',
                'tiff_dir': Path,
                'segmented_dir': list[Path] | None.
                'morphometrics_dir': Path | None,
                'csa_dir': Path | None,
            }
        }
    }
    Ignores 4X_TIFF folders if ignore_4x=True.
    """
    config        = load_config()
    tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
    seg_folder    = config.get("segmented_folder",    "Segmented")
    morph_folder  = config.get("morphometrics_folder","Morphometrics")
    csa_folder    = config.get("csa_folder",          "CSA")

    study_path = Path(study_dir)
    result     = {}

    for animal_path in sorted(study_path.iterdir()):
        if not animal_path.is_dir():
            continue
        animal_name         = animal_path.name
        result[animal_name] = {}

        try:
            nerve_paths = sorted(animal_path.iterdir())
        except PermissionError:
            continue
        for nerve_path in nerve_paths:
            if not nerve_path.is_dir():
                continue
            nerve_name = nerve_path.name

            mag      = None
            tiff_dir = None
            for sub in nerve_path.iterdir():
                matched_suffix = next(
                    (s for s in tiff_suffixes if sub.name.upper().endswith(s)), None
                )
                if sub.is_dir() and matched_suffix:
                    detected_mag = _strip_suffix(sub.name.upper(), matched_suffix)
                    if ignore_4x and detected_mag == "4X":
                        continue
                    mag      = detected_mag
                    tiff_dir = sub
                    break

            if tiff_dir is None:
                continue
            
            seg_candidates = [d for d in nerve_path.iterdir() 
                  if d.is_dir() and d.name.startswith(seg_folder)]
            
            seg_dirs = [d for d in seg_candidates if _has_images(d, tiff_dir)]
            if not seg_dirs:
                seg_dirs = None
            morph_dir = nerve_path / morph_folder
            csa_dir   = nerve_path / csa_folder

            result[animal_name][nerve_name] = {
                'mag':               mag,
                'tiff_dir':          tiff_dir,
                'segmented_dir':     seg_dirs,
                'morphometrics_dir': morph_dir if morph_dir.exists()             else None,
                'csa_dir':           csa_dir   if csa_dir.exists()               else None,
            }

    return result


def _scan_animal_nerves(animal_path: Path, ignore_4x: bool = True) -> dict:
    """
    Scan a single animal directory's nerves directly, without assuming a
    study-level root above it. Used as resolve_scan()'s PermissionError
    fallback for 'animal'-level input — calling scan_study() on an animal
    path instead of a study path misreads the directory depth (nerve
    folders get treated as animals) and silently returns an empty result.
    
    ignore_4x mirrors scan_study()'s parameter of the same name.
    """
    config        = load_config()
    tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
    seg_folder    = config.get("segmented_folder",    "Segmented")
    morph_folder  = config.get("morphometrics_folder","Morphometrics")
    csa_folder    = config.get("csa_folder",          "CSA")

    result = {}
    for nerve_path in sorted(animal_path.iterdir()):
        if not nerve_path.is_dir():
            continue
        mag, tiff_dir = None, None
        for sub in nerve_path.iterdir():
            matched_suffix = next((s for s in tiff_suffixes if sub.name.upper().endswith(s)), None)
            if sub.is_dir() and matched_suffix:
                detected_mag = _strip_suffix(sub.name.upper(), matched_suffix)
                if ignore_4x and detected_mag == "4X":
                    continue
                mag, tiff_dir = detected_mag, sub
                break
        if tiff_dir is None:
            continue
        seg_candidates = [d for d in nerve_path.iterdir() if d.is_dir() and d.name.startswith(seg_folder)]
        seg_dirs = [d for d in seg_candidates if _has_images(d, tiff_dir)] or None
        morph_dir = nerve_path / morph_folder
        csa_dir   = nerve_path / csa_folder
        result[nerve_path.name] = {
            'mag': mag, 'tiff_dir': tiff_dir, 'segmented_dir': seg_dirs,
            'morphometrics_dir': morph_dir if morph_dir.exists() else None,
            'csa_dir': csa_dir if csa_dir.exists() else None,
        }
    return result


def detect_input_level(path: str) -> str:
    """
    Detect whether a path is at study, animal, nerve, or tiff level.
    Returns: 'study', 'animal', 'nerve', or 'tiff'
    """
    config        = load_config()
    tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
    p             = Path(path)


    def has_tiff(folder):
        return any(
            sub.is_dir() and any(sub.name.upper().endswith(s) for s in tiff_suffixes)
            for sub in folder.iterdir()
        )

    if any(p.name.upper().endswith(s) for s in tiff_suffixes):
        return 'tiff'
    if has_tiff(p):
        return 'nerve'
    subdirs = [s for s in p.iterdir() if s.is_dir()]
    if subdirs and any(has_tiff(s) for s in subdirs):
        return 'animal'
    return 'study'


def resolve_scan(input_dir: str, *, ignore_4x: bool = True) -> tuple:
    """
    Accepts study, animal, nerve, or tiff level input.
    Returns (study_dict, study_dir) in the same format scan_study produces.
    
    ignore_4x mirrors scan_study()'s own parameter — it now applies
    consistently across every level, not just the 'study' and 'animal'
    levels that happened to delegate to scan_study() directly.
    """
    level  = detect_input_level(input_dir)
    p      = Path(input_dir)
    config = load_config()

    reserved = {
        config.get("segmented_folder",    "Segmented").lower(),
        config.get("morphometrics_folder","Morphometrics").lower(),
        config.get("csa_folder",          "CSA").lower(),
    }
    if p.name.lower() in reserved:
        raise ValueError(
            f"'{p.name}' is an output subfolder — please pass the nerve, animal, or study folder instead."
        )

    if level == 'tiff':
        nerve_path    = p.parent
        animal_path   = nerve_path.parent
        study_dir     = str(animal_path.parent)
        seg_folder    = config.get("segmented_folder",    "Segmented")
        morph_folder  = config.get("morphometrics_folder","Morphometrics")
        csa_folder    = config.get("csa_folder",          "CSA")
        tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
        matched       = next((s for s in tiff_suffixes if p.name.upper().endswith(s)), "_TIFF")
        mag           = _strip_suffix(p.name.upper(), matched)
        nerve_data    = {
            'mag':               mag,
            'tiff_dir':          p,
            'segmented_dir':     None,
            'morphometrics_dir': p.parent / morph_folder if (p.parent / morph_folder).exists() else None,
            'csa_dir':           p.parent / csa_folder   if (p.parent / csa_folder).exists()   else None,
        }
        return {animal_path.name: {nerve_path.name: nerve_data}}, study_dir

    if level == 'nerve':
        animal_path  = p.parent
        study_dir    = str(animal_path.parent)

        # Guard — build nerve data directly rather than scanning upward.
        # Scanning upward from a shallow path (e.g. Desktop) risks iterating
        # protected system folders. Direct construction is always safe.
        seg_folder    = config.get("segmented_folder",    "Segmented")
        morph_folder  = config.get("morphometrics_folder","Morphometrics")
        csa_folder    = config.get("csa_folder",          "CSA")
        tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]

        tiff_dir = next(
            (sub for sub in sorted(p.iterdir())
             if sub.is_dir()
             and any(sub.name.upper().endswith(s) for s in tiff_suffixes)
             and (not ignore_4x or _strip_suffix(
                 sub.name.upper(),
                 next((s for s in tiff_suffixes if sub.name.upper().endswith(s)), "")
             ) != "4X")),
            None
        )
        matched  = next(
            (s for s in tiff_suffixes if tiff_dir and tiff_dir.name.upper().endswith(s)),
            "_TIFF"
        ) if tiff_dir else "_TIFF"
        mag      = _strip_suffix(tiff_dir.name.upper(), matched) if tiff_dir else None

        seg_candidates = [
            d for d in p.iterdir()
            if d.is_dir() and d.name.startswith(seg_folder)
        ] if tiff_dir else []
        seg_dirs = [d for d in seg_candidates if _has_images(d, tiff_dir)] or None

        nerve_data = {
            'mag':               mag,
            'tiff_dir':          tiff_dir,
            'segmented_dir':     seg_dirs,
            'morphometrics_dir': (p / morph_folder) if (p / morph_folder).exists() else None,
            'csa_dir':           (p / csa_folder)   if (p / csa_folder).exists()   else None,
        }
        return {animal_path.name: {p.name: nerve_data}}, study_dir

    elif level == 'animal':
        study_dir = str(p.parent)
        try:                                                           
            full = scan_study(study_dir, ignore_4x=ignore_4x)
            animal_data = full.get(p.name, {})                              
        except PermissionError:                                       
            animal_data = _scan_animal_nerves(p, ignore_4x=ignore_4x)
        return {p.name: animal_data}, study_dir

    else:
        return scan_study(input_dir, ignore_4x=ignore_4x), input_dir


def detect_study_mag(study_result: dict) -> str | None:
    """Detect the dominant magnification across a study (ignoring 4X)."""
    mags = [
        nerve.get('mag')
        for animal in study_result.values()
        for nerve in animal.values()
        if nerve.get('mag')
    ]
    if not mags:
        return None
    return max(set(mags), key=mags.count)