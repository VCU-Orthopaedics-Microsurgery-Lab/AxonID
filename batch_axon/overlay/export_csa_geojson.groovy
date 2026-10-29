/**
 * batch_axon/overlay/export_csa_geojson.groovy
 *
 * QuPath → Batch_Axon integration script.
 *
 * Exports "+" / "-" CSA annotations from qualifying QuPath project images
 * as GeoJSON FeatureCollections for processing by AxonID.
 *
 * Output:
 *     <NerveFolder>/CSA/<image_stem>_CSA.geojson
 *
 * Supports annotations identified through either the Name field or
 * Classification / Set class.
 *
 * EXCLUDING A WHOLE IMAGE
 * Give any annotation on the image (any shape, a small box is enough) the
 * classification "EXCLUDE" (any capitalisation) and write the REASON in the
 * annotation's Name or its Description. AxonID then leaves the whole image
 * out of every count, mean, distribution and total, and lists it with the
 * reason. A reason is required: an EXCLUDE annotation with neither Name nor
 * Description is reported under ANNOTATIONS NEEDING REVIEW and is NOT
 * exported, so the image is not excluded. The reason is written to the
 * GeoJSON as the feature's name, so it reaches AxonID whichever field you
 * used. The image's "+" / "-" annotations are not needed, but may stay.
 * Only 40X images use this tag.
 *
 * The NerveFolder is recovered from each image's stored file path rather
 * than hardcoded. Images without qualifying CSA annotations are skipped
 * and reported in the final summary.
 *
 * ANNOTATIONS NEEDING REVIEW
 * The summary opens with an annotation-level review count: every
 * annotation that was NOT exported and needs user attention. It is the sum
 * of three disjoint categories, each listed in detail below the count:
 *     untagged   — annotations with no "+"/"-" name or class, including
 *                  whitespace near-misses (" +", "- ") that fail the
 *                  exact-match test
 *     open-path  — tagged "+"/"-" but not an enclosed area
 *     zero-area  — tagged "+"/"-" but area <= ZERO_AREA_EPS
 * Image-level skips (no +/- annotations, file not found, bad path) are
 * counted separately and listed last. Images skipped for "file not found"
 * or "bad path" are never opened, so their annotations are not counted.
 *
 * DATA QUALITY CHECKS
 * Open-path annotations (roi.isArea() == false — Polyline objects, or a
 * Polygon whose endpoints were never joined) are excluded from export
 * and reported separately, split by inclusion ('+') vs exclusion ('-').
 * Polyline is a distinct QuPath object type with no enclosed area by
 * design — overlapping its endpoints does NOT fix it; delete and
 * re-trace with the Polygon or Brush tool.
 *
 * Zero-area annotations (area <= ZERO_AREA_EPS) that ARE proper area-type
 * ROIs are also excluded — almost always a stray click.
 *
 * Untagged annotations are reported but never exported. Open-path
 * untagged annotations are listed separately from closed-area ones.
 *
 * A commented-out area-precision diagnostic is kept in this script
 * (search "irregular"). It is not needed for normal use.
 *
 * IDENTIFYING AN ANNOTATION IN THE GUI
 * Every matched annotation prints both raw pixel area and calibrated
 * um^2 area (this image's own pixel calibration — same source as
 * QuPath's Annotation panel). Match by that value, not list position.
 *
 * DRY_RUN (below) is true by default: the script reports what it would
 * export without writing files. Set it to false to write the GeoJSONs.
 *
 * Run via: Run -> Run with the target cohort's QuPath project open.
 * Do NOT use "Run for project": the script already loops over every image
 * in the open project, so that option would repeat the full loop once per
 * image.
 */

def DRY_RUN = true
def ZERO_AREA_EPS = 0.001

import qupath.lib.objects.PathAnnotationObject
// import org.locationtech.jts.geom.Polygon as JTSPolygon   // only needed by the commented-out irregular-precision diagnostic below
import java.nio.file.Paths

def project = getProject()
if (project == null) {
    print "ERROR: no project open."
    return
}

// Short path for display only: "Cohort_1\Animal\Nerve\CSA\file.geojson".
// Actual read/write paths are always the full path.
def shortPath = { study, animal, nerve, filename ->
    def s = study?.getFileName()?.toString() ?: "?"
    def a = animal?.getFileName()?.toString() ?: "?"
    def n = nerve?.getFileName()?.toString() ?: "?"
    return "${s}\\${a}\\${n}\\CSA\\${filename}"
}

// Trim a double to N decimals for display only.
def fmt = { Double v, int n = 4 -> v == null ? "?" : String.format("%.${n}f", v) }

// // Shoelace area of one closed JTS ring — only used by the commented-out
// // irregular-precision diagnostic below. See file header for context.
// def shoelaceAreaOfRing = { coords ->
//     int n = coords.size() - 1
//     double s = 0
//     for (int i = 0; i < n; i++) {
//         def p1 = coords[i]
//         def p2 = coords[(i + 1) % n]
//         s += p1.x * p2.y - p2.x * p1.y
//     }
//     return Math.abs(s) / 2.0
// }
//
// def countNonInt = { coords ->
//     coords.count { c -> Math.abs(c.x - Math.round(c.x)) > 1e-6 || Math.abs(c.y - Math.round(c.y)) > 1e-6 }
// }

int exported = 0
int skipped  = 0
def skippedByReason = [:].withDefault { [] }

// Whole-image exclusion tag (see header): class "EXCLUDE", reason in Name or Description.
def isExclude = { a -> (a.getPathClass()?.getName() ?: "").trim().equalsIgnoreCase("EXCLUDE") }
def reasonOf = { a ->
    def n = (a.getName() ?: "").trim()
    if (n) return n
    def d = null
    try { d = a.getDescription() } catch (Throwable t) { d = a.getMetadata()?.get("description") }
    return (d ?: "").toString().trim()
}
def excludedImages        = []     // [image, reason]
def excludeNoReasonList   = []     // [image, roiType]


def zeroAreaList       = []
def openPathList       = []
def untaggedList       = []
def openPathTypeCounts = [:].withDefault { 0 }
// def irregularList    = []   // commented out with the diagnostic below

for (entry in project.getImageList()) {
    def imageName = entry.getImageName()

    def uris = entry.getServerBuilder().getURIs()
    if (uris.isEmpty()) {
        print "SKIP (file not found): ${imageName}"
        skipped++
        skippedByReason["Original file not found"] << imageName
        continue
    }
    def imagePath = Paths.get(uris[0])
    def imageStem = imagePath.getFileName().toString().replaceFirst(/\.ome\.tif(f)?$|\.tif(f)?$/, '')

    // .../Cohort/Animal/Nerve/40X_TIFF/image.tif
    def magFolder    = imagePath.getParent()
    def nerveFolder  = magFolder?.getParent()
    def animalFolder = nerveFolder?.getParent()
    def studyFolder  = animalFolder?.getParent()

    if (nerveFolder == null) {
        print "SKIP (bad path): ${imageName}"
        skipped++
        skippedByReason["Couldn't resolve Nerve folder"] << imageName
        continue
    }

    def imageData = entry.readImageData()
    def cal = imageData.getServer()?.getPixelCalibration()
    def pxW = cal?.getPixelWidthMicrons()
    def pxH = cal?.getPixelHeightMicrons()

    def rawMatches = imageData.getHierarchy().getAnnotationObjects().findAll {
        def name = it.getName()
        def className = it.getPathClass()?.getName()
        def matchesName = (name == "+" || name == "-")
        def matchesClass = (className == "+" || className == "-")
        if (matchesName && matchesClass && name != className) {
            print "WARN name/class differ: ${imageName}"
            print "  name='${name}' class='${className}', using name"
        }
        matchesName || matchesClass
    }

    def excludeValid = []
    imageData.getHierarchy().getAnnotationObjects().findAll { isExclude(it) }.each { a ->
        def why = reasonOf(a)
        if (why) {
            excludeValid << [ann: a, reason: why]
        } else {
            excludeNoReasonList << [image: imageName, roiType: a.getROI()?.getClass()?.getSimpleName() ?: "null"]
            print "  EXCLUDE tag WITHOUT a reason (Name or Description): not exported, image NOT excluded"
        }
    }
    if (!excludeValid.isEmpty() && magFolder?.getFileName()?.toString()?.toUpperCase()?.startsWith("4X")) {
        print "  WARN: EXCLUDE on a 4X image is not used by AxonID (only 40X images)"
    }

    def untagged = imageData.getHierarchy().getAnnotationObjects().findAll { !rawMatches.contains(it) && !isExclude(it) }
    untagged.each { u ->
        def r       = u.getROI()
        def nm      = u.getName()
        def cn      = u.getPathClass()?.getName()
        def rt      = r?.getClass()?.getSimpleName() ?: "null"
        def aPx     = r?.getArea()
        def aUm2    = (aPx != null && pxW && pxH) ? aPx * pxW * pxH : null
        // whitespace near-miss: " +", "+ ", "- " etc. — silently unmatched by the exact == test
        def nearMiss = (nm != null && nm != nm.trim() && nm.trim() in ["+", "-"]) ||
                    (cn != null && cn != cn.trim() && cn.trim() in ["+", "-"])
        untaggedList << [image: imageName, name: nm, cls: cn, roiType: rt,
                        isArea: r?.isArea(), areaPx: aPx, areaUm2: aUm2, nearMiss: nearMiss]
        print "  UNTAGGED [name=${nm ?: '-'} class=${cn ?: '-'}] ${rt} px2=${fmt(aPx)} um2=${fmt(aUm2)}${nearMiss ? '  <-- WHITESPACE NEAR-MISS' : ''}"
    }

    if (rawMatches.isEmpty() && excludeValid.isEmpty()) {
        print "SKIP (no +/-): ${imageName}"
        skipped++
        skippedByReason["No +/- annotations"] << imageName
        continue
    }


    def annotations = []
    for (a in rawMatches) {
        def roi  = a.getROI()
        def area = roi?.getArea()
        def cls  = a.getPathClass()?.getName()

        if (roi != null && !roi.isArea()) {
            def roiType = roi.getClass().getSimpleName()
            openPathTypeCounts[roiType]++
            openPathList << [image: imageName, cls: cls, roiType: roiType]
            print "  EXCLUDED open-path [${cls}]"
        } else if (area == null || area <= ZERO_AREA_EPS) {
            zeroAreaList << [image: imageName, cls: cls, areaPx: area]
            print "  EXCLUDED zero-area [${cls}]"
        } else {
            annotations << a
        }
    }

    if (annotations.isEmpty() && excludeValid.isEmpty()) {
        print "SKIP (nothing left): ${imageName}"
        skipped++
        skippedByReason["All annotations open-path or zero-area"] << imageName
        continue
    }

    annotations.each { annotation ->
        def roi = annotation.getROI()
        def area = roi?.getArea()
        def areaUm2 = (area != null && pxW && pxH) ? area * pxW * pxH : null
        def cls = annotation.getPathClass()?.getName()

        // // --- IRREGULAR-PRECISION DIAGNOSTIC (resolved, see file header) ---
        // // Uncomment this block, plus irregularList, the two closures, and
        // // the JTSPolygon import above, and the summary section below, to
        // // re-run this investigation on a future cohort if ever needed.
        // def isIrregular = area != null && Math.abs((area * 2) - Math.round(area * 2)) > 1e-6
        // if (isIrregular) {
        //     def roiType = roi?.getClass()?.getSimpleName()
        //     def geom = roi?.getGeometry()
        //     def geomClass = geom?.getClass()?.getSimpleName()
        //     def geomArea = geom?.getArea()
        //     def numSubGeoms = geom?.getNumGeometries() ?: 0
        //     def ringReports = []
        //     double manualArea = 0.0
        //     def sample = null
        //     for (int gi = 0; gi < numSubGeoms; gi++) {
        //         def sub = geom.getGeometryN(gi)
        //         if (!(sub instanceof JTSPolygon)) {
        //             ringReports << "sub${gi}: NOT a Polygon (${sub?.getClass()?.getSimpleName()})"
        //             continue
        //         }
        //         def poly = (JTSPolygon) sub
        //         def shell = poly.getExteriorRing()
        //         def shellCoords = shell.getCoordinates()
        //         manualArea += shoelaceAreaOfRing(shellCoords)
        //         ringReports << "sub${gi} shell pts=${shellCoords.size()} nonInt=${countNonInt(shellCoords)}"
        //         if (sample == null) {
        //             sample = shellCoords.take(5).collect { "(${it.x as int},${it.y as int})" }.join(' ')
        //         }
        //         int nHoles = poly.getNumInteriorRing()
        //         for (int hi = 0; hi < nHoles; hi++) {
        //             def hole = poly.getInteriorRingN(hi)
        //             def hc = hole.getCoordinates()
        //             manualArea -= shoelaceAreaOfRing(hc)
        //             ringReports << "sub${gi} hole${hi} pts=${hc.size()} nonInt=${countNonInt(hc)}"
        //         }
        //     }
        //     irregularList << [
        //         image: imageName, cls: cls, areaPx: area, areaUm2: areaUm2,
        //         roiType: roiType, geomClass: geomClass, numSubGeoms: numSubGeoms,
        //         geomArea: geomArea, manualArea: manualArea,
        //         diffRoiGeom:    (geomArea != null) ? (area - geomArea) : null,
        //         diffRoiManual:  (area - manualArea),
        //         diffGeomManual: (geomArea != null) ? (geomArea - manualArea) : null,
        //         ringReports: ringReports, sample: sample
        //     ]
        // }
        // --- end diagnostic ---

        def nameTag = annotation.getName() ? " ${annotation.getName()}" : ""
        print "  [${cls}]${nameTag} px2=${fmt(area)} um2=${fmt(areaUm2)}"
    }

    // The exclusion annotation is exported as a copy whose NAME is the reason, so the
    // GeoJSON carries it whether it was typed as Name or as Description.
    def exportList = new ArrayList(annotations)
    excludeValid.each { e ->
        def copy = new PathAnnotationObject(e.ann.getROI(), e.ann.getPathClass())
        copy.setName(e.reason)
        exportList << copy
        excludedImages << [image: imageName, reason: e.reason]
        print "  [EXCLUDE] ${e.reason}"
    }

    def csaDir = nerveFolder.resolve("CSA")
    def displayPath = shortPath(studyFolder, animalFolder, nerveFolder, "${imageStem}_CSA.geojson")

    if (DRY_RUN) {
        print "DRY RUN: ${exportList.size()} ann"
        print "  -> ${displayPath}"
        exported++
        continue
    }

    csaDir.toFile().mkdirs()
    def outPath = csaDir.resolve("${imageStem}_CSA.geojson")
    exportObjectsToGeoJson(exportList, outPath.toString(), "FEATURE_COLLECTION")
    print "Wrote: ${exportList.size()} ann"
    print "  -> ${displayPath}"
    exported++
}

print ""
print "=" * 40
print "SUMMARY"
print "=" * 40

int nearMissCount = untaggedList.count { it.nearMiss }
int openInclCount = openPathList.count { it.cls == "+" }
int needsReview   = untaggedList.size() + openPathList.size() + zeroAreaList.size() + excludeNoReasonList.size()

if (needsReview == 0) {
    print "Annotations needing review: 0"
} else {
    print "ANNOTATIONS NEEDING REVIEW: ${needsReview}"
    print "  untagged:   ${untaggedList.size()}  (whitespace near-misses: ${nearMissCount})"
    print "  open-path:  ${openPathList.size()}  (inclusions '+': ${openInclCount})"
    print "  zero-area:  ${zeroAreaList.size()}"
    print "  EXCLUDE without reason: ${excludeNoReasonList.size()}"
}
print ""
print "Images exported: ${exported}  |  Images skipped: ${skipped}"


if (!excludedImages.isEmpty()) {
    print ""
    print "IMAGES MARKED EXCLUDED (${excludedImages.size()}), left out of every AxonID number:"
    excludedImages.each { print "  ${it.image}: ${it.reason}" }
}

if (!excludeNoReasonList.isEmpty()) {
    print ""
    print "EXCLUDE tags WITHOUT A REASON (${excludeNoReasonList.size()}), NOT exported; type the reason in Name or Description:"
    excludeNoReasonList.each { print "  ${it.image} (${it.roiType})" }
}

if (!openPathList.isEmpty()) {
    print ""
    print "OPEN-PATH annotations excluded (${openPathList.size()}):"
    print "  Polyline has no enclosed area by design."
    print "  Overlapping its endpoints will NOT fix it —"
    print "  delete and re-trace with Polygon or Brush."
    def incl = openPathList.findAll { it.cls == "+" }
    def excl = openPathList.findAll { it.cls != "+" }
    if (!incl.isEmpty()) {
        print "  INCLUSIONS (high priority — loses sampled tissue):"
        incl.each { print "    ${it.image} (${it.roiType})" }
    }
    if (!excl.isEmpty()) {
        print "  Exclusions:"
        excl.each { print "    ${it.image} (${it.roiType})" }
    }
    print "  By type:"
    openPathTypeCounts.each { t, c -> print "    ${t}: ${c}" }
}

if (!zeroAreaList.isEmpty()) {
    print ""
    print "ZERO-AREA annotations excluded (${zeroAreaList.size()}):"
    zeroAreaList.each { print "  ${it.image} [${it.cls}]" }
}

// if (!irregularList.isEmpty()) {
//     print ""
//     print "IRREGULAR-PRECISION annotations (${irregularList.size()}), not excluded:"
//     print "  Area triangulated 3 ways per annotation: roi.getArea(),"
//     print "  geom.getArea(), and an independent shoelace recompute"
//     print "  over every ring of every sub-geometry found."
//     irregularList.each { r ->
//         print "  ${r.image} [${r.cls}]"
//         print "    roi.getArea()   = ${r.areaPx}"
//         print "    geom.getArea()  = ${r.geomArea}"
//         print "    manual shoelace = ${r.manualArea}"
//         print "    diff roi-geom   = ${r.diffRoiGeom}"
//         print "    diff roi-manual = ${r.diffRoiManual}"
//         print "    diff geom-manual= ${r.diffGeomManual}"
//         print "    geomClass=${r.geomClass}  subGeoms=${r.numSubGeoms}"
//         r.ringReports.each { print "    ${it}" }
//         print "    sample: ${r.sample}"
//     }
// }

if (!untaggedList.isEmpty()) {
    print ""
    print "UNTAGGED annotations (${untaggedList.size()}), NOT exported:"
    def nm = untaggedList.findAll { it.nearMiss }
    if (!nm.isEmpty()) {
        print "  WHITESPACE NEAR-MISSES (high priority — almost certainly meant as +/-):"
        nm.each { print "    ${it.image} name='${it.name}' class='${it.cls}' (${it.roiType})" }
    }
    def open = untaggedList.findAll { !it.nearMiss && it.isArea == false }
    if (!open.isEmpty()) {
        print "  OPEN-PATH (Polyline etc.):"
        open.each { print "    ${it.image} (${it.roiType}) name=${it.name ?: '-'} class=${it.cls ?: '-'}" }
    }
    def closed = untaggedList.findAll { !it.nearMiss && it.isArea != false }
    if (!closed.isEmpty()) {
        print "  CLOSED-AREA:"
        closed.each { print "    ${it.image} (${it.roiType}) um2=${fmt(it.areaUm2)} name=${it.name ?: '-'} class=${it.cls ?: '-'}" }
    }
    print "  By image:"
    untaggedList.countBy { it.image }.each { img, c -> print "    ${img}: ${c}" }
}

if (!skippedByReason.isEmpty()) {
    print ""
    print "IMAGES SKIPPED, by reason:"
    skippedByReason.each { reason, images ->
        print "  ${reason} (${images.size()}):"
        images.each { print "    ${it}" }
    }
}

print "=" * 40