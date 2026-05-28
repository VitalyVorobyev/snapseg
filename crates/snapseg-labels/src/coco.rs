//! COCO-shaped export of snapseg labels.
//!
//! Maps each `labels/<id>/` directory to one COCO image + one COCO
//! annotation. Binary masks are encoded as **uncompressed COCO RLE**
//! (`counts` as a `Vec<u32>` of alternating background/foreground
//! run-lengths walked column-major, starting with the background run).
//! Polygon segmentations and compressed RLE land when subpixel edges
//! (M3) do.
//!
//! The current export emits a single category — `id = 1`, name
//! `"foreground"` — because snapseg labels are binary (one
//! foreground region per label) per [`AGENTS.md`]. Multi-class
//! semantic segmentation is an explicit non-goal.
//!
//! [`AGENTS.md`]: ../../AGENTS.md

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{LabelError, LabelQuality, PolygonJson, Provenance, SCHEMA_VERSION};

/// Options controlling the conversion.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Drop labels whose `operator.quality == Reject` from the export.
    /// Defaults to `true`; the rejected-as-good label is more harmful
    /// than the dropped-good-label is.
    pub skip_rejected: bool,
    /// When `true`, the first per-label error aborts conversion.
    /// When `false` (the default), per-label errors are logged via
    /// `tracing::warn` and the bad label is skipped.
    pub strict: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            skip_rejected: true,
            strict: false,
        }
    }
}

/// Result of a successful [`convert_dir`] call.
#[derive(Debug)]
pub struct ConversionReport {
    /// The assembled COCO dataset.
    pub dataset: CocoDataset,
    /// Number of label directories that errored and were skipped.
    /// Always `0` in strict mode (any error aborts conversion).
    pub skipped: usize,
}

/// Top-level COCO dataset. Matches the canonical
/// `info` / `images` / `annotations` / `categories` shape.
#[derive(Debug, Serialize, Deserialize)]
pub struct CocoDataset {
    /// Dataset-level metadata.
    pub info: CocoInfo,
    /// One [`CocoImage`] per included label.
    pub images: Vec<CocoImage>,
    /// One [`CocoAnnotation`] per included label (1:1 with images).
    pub annotations: Vec<CocoAnnotation>,
    /// Always a single-entry foreground category in this version.
    pub categories: Vec<CocoCategory>,
}

/// Dataset-level metadata.
#[derive(Debug, Serialize, Deserialize)]
pub struct CocoInfo {
    /// Human-readable dataset description.
    pub description: String,
    /// Dataset version string. Snapseg writes `"1.0"` today; bump as
    /// downstream consumers diverge.
    pub version: String,
    /// On-disk schema version of the labels this dataset was rolled
    /// from. Matches [`crate::SCHEMA_VERSION`].
    pub schema_version: String,
    /// Calendar year, taken from the system clock at export time.
    pub year: u32,
}

/// One COCO image entry. Corresponds to one snapseg label directory.
#[derive(Debug, Serialize, Deserialize)]
pub struct CocoImage {
    /// Monotonically increasing identifier, 1-based.
    pub id: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Path relative to the labels/ root, e.g. `"abc12345/image.png"`.
    pub file_name: String,
    /// RFC3339 timestamp from the label's `meta.toml`.
    pub date_captured: String,
    /// Human-readable note: `"<registry_name> · <quality> · <commit?>"`.
    pub note: String,
    /// Full snapseg [`Provenance`] for downstream filtering by family,
    /// runtime, operator quality, commit, …
    pub snapseg_provenance: Provenance,
}

/// One COCO annotation entry. snapseg writes one annotation per image.
#[derive(Debug, Serialize, Deserialize)]
pub struct CocoAnnotation {
    /// Monotonically increasing identifier, 1-based.
    pub id: u64,
    /// Owning [`CocoImage::id`].
    pub image_id: u64,
    /// Always `1` (foreground); see module docs.
    pub category_id: u32,
    /// `[x, y, w, h]` of the mask's tight axis-aligned bounding box.
    pub bbox: [f64; 4],
    /// Foreground pixel count.
    pub area: f64,
    /// Mask segmentation; uncompressed RLE today.
    pub segmentation: CocoSegmentation,
    /// Always `0`; snapseg labels are single-instance.
    pub iscrowd: u32,
}

/// Polygon or uncompressed RLE segmentation. COCO consumers expect a
/// JSON array of arrays for polygons and a `{counts, size}` object for
/// RLE — `serde(untagged)` lets both shapes coexist on the same field.
///
/// Variant ordering matters here: `Polygon` is listed first so a JSON
/// array deserializes as `Polygon` rather than getting misclassified.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CocoSegmentation {
    /// Polygon segmentation. Outer Vec for multiple polygons (always
    /// one element today); inner Vec is interleaved `[x1, y1, x2, y2,
    /// …]` in source-image pixel coordinates.
    Polygon(Vec<Vec<f64>>),
    /// Uncompressed column-major RLE.
    UncompressedRle {
        /// Alternating background/foreground run-lengths starting from
        /// background. Walked column-major. If the first pixel is
        /// foreground, the first entry is a leading zero.
        counts: Vec<u32>,
        /// `[height, width]` of the mask.
        size: [u32; 2],
    },
}

/// One COCO category. snapseg emits exactly one — `id = 1`, name
/// `"foreground"` — because snapseg labels are binary.
#[derive(Debug, Serialize, Deserialize)]
pub struct CocoCategory {
    /// Category id; always `1` in this version.
    pub id: u32,
    /// Human-readable category name.
    pub name: String,
    /// Parent category name; `"snapseg"`.
    pub supercategory: String,
}

/// Convert every label directory under `label_root` into a
/// [`ConversionReport`] containing the assembled [`CocoDataset`] and
/// the count of labels skipped due to errors.
///
/// Iteration is sorted by directory name so the output is
/// deterministic across runs. Directory entries whose name ends with
/// `.partial` are skipped (these are in-progress writes by
/// [`crate::LabelDir::save`]).
///
/// `image_id` and `annotation_id` start at `1` and increment in
/// lockstep — one image per annotation for now.
///
/// When [`ConvertOptions::strict`] is `false` (the default), a
/// per-label parse or I/O error is logged via `tracing::warn` and
/// the label is skipped; the returned `skipped` count reflects how
/// many were dropped this way. When `strict` is `true`, the first
/// per-label error aborts conversion and propagates as `Err`.
///
/// # Errors
///
/// - [`LabelError::Io`] if `label_root` cannot be read or a label
///   directory cannot be opened.
/// - Per-label [`LabelError::TomlDe`] / [`LabelError::Json`] /
///   [`LabelError::Image`] / [`LabelError::MissingArtifact`] /
///   [`LabelError::MaskFormat`] only propagate in strict mode; in
///   lenient mode they are logged and counted in `skipped`.
pub fn convert_dir(
    label_root: &Path,
    opts: ConvertOptions,
) -> Result<ConversionReport, LabelError> {
    let entries = sorted_label_dirs(label_root)?;

    let mut images: Vec<CocoImage> = Vec::new();
    let mut annotations: Vec<CocoAnnotation> = Vec::new();
    let mut next_id: u64 = 1;
    let mut skipped: usize = 0;

    for dir in entries {
        // skip_rejected is not an error; it does not count toward skipped.
        match read_meta(&dir) {
            Ok(meta)
                if opts.skip_rejected && matches!(meta.operator.quality, LabelQuality::Reject) =>
            {
                continue;
            }
            Ok(meta) => match per_label_work(&dir, meta, next_id) {
                Ok((image, annotation)) => {
                    images.push(image);
                    annotations.push(annotation);
                    next_id += 1;
                }
                Err(e) => {
                    if opts.strict {
                        return Err(e);
                    }
                    tracing::warn!(label = %dir.display(), error = %e, "skipping malformed label");
                    skipped += 1;
                }
            },
            Err(e) => {
                if opts.strict {
                    return Err(e);
                }
                tracing::warn!(label = %dir.display(), error = %e, "skipping malformed label");
                skipped += 1;
            }
        }
    }

    Ok(ConversionReport {
        dataset: CocoDataset {
            info: CocoInfo {
                description: "snapseg labels exported to COCO".to_string(),
                version: "1.0".to_string(),
                schema_version: SCHEMA_VERSION.to_string(),
                year: chrono::Utc::now()
                    .format("%Y")
                    .to_string()
                    .parse::<u32>()
                    .unwrap_or(0),
            },
            images,
            annotations,
            categories: vec![CocoCategory {
                id: 1,
                name: "foreground".to_string(),
                supercategory: "snapseg".to_string(),
            }],
        },
        skipped,
    })
}

/// Process one label directory into a `(CocoImage, CocoAnnotation)` pair.
///
/// Reads and validates all label artifacts; returns `Err` on any
/// parse, I/O, or dimension mismatch. The caller decides whether to
/// propagate or skip based on [`ConvertOptions::strict`].
fn per_label_work(
    dir: &Path,
    meta: Provenance,
    id: u64,
) -> Result<(CocoImage, CocoAnnotation), LabelError> {
    label_to_coco_entries(dir, meta, id, id)
}

/// Convert + write `label_root` to `out_path` as pretty JSON.
///
/// Returns a [`ConversionReport`] containing the dataset and the count
/// of labels skipped due to errors (see [`convert_dir`]).
///
/// # Errors
///
/// Same set as [`convert_dir`], plus [`LabelError::Io`] /
/// [`LabelError::Json`] for the output write.
pub fn convert_dir_to_file(
    label_root: &Path,
    out_path: &Path,
    opts: ConvertOptions,
) -> Result<ConversionReport, LabelError> {
    let report = convert_dir(label_root, opts)?;
    let json = serde_json::to_string_pretty(&report.dataset)?;
    fs::write(out_path, json)?;
    Ok(report)
}

/// List every plausible per-label subdirectory under `label_root`,
/// sorted ascending by file name.
///
/// Skips files, `.partial` orphans (in-progress writes), and entries
/// whose name starts with `.` (hidden dotfiles like `.DS_Store`).
fn sorted_label_dirs(label_root: &Path) -> Result<Vec<PathBuf>, LabelError> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(label_root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy().to_string();
        if name.starts_with('.') || name.ends_with(".partial") {
            continue;
        }
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        dirs.push(path);
    }
    dirs.sort();
    Ok(dirs)
}

/// Read and parse `<label_dir>/meta.toml`.
fn read_meta(label_dir: &Path) -> Result<Provenance, LabelError> {
    let meta_path = label_dir.join("meta.toml");
    if !meta_path.exists() {
        return Err(LabelError::MissingArtifact {
            label: label_id_from_dir(label_dir),
            artifact: "meta.toml",
        });
    }
    let s = fs::read_to_string(&meta_path)?;
    let prov: Provenance = toml::from_str(&s)?;
    Ok(prov)
}

// why this is long: builds one COCO image+annotation pair, with two
// segmentation paths (polygon when polygon.json is present, RLE fallback
// from mask.png otherwise) plus the dimension / artifact checks that
// must run in order before we choose between them. Splitting would
// scatter the order across helpers and obscure the fallback logic.
fn label_to_coco_entries(
    label_dir: &Path,
    meta: Provenance,
    image_id: u64,
    annotation_id: u64,
) -> Result<(CocoImage, CocoAnnotation), LabelError> {
    let id = label_id_from_dir(label_dir);

    // image.png — present-check only; we don't need its bytes here.
    let image_path = label_dir.join("image.png");
    if !image_path.exists() {
        return Err(LabelError::MissingArtifact {
            label: id,
            artifact: "image.png",
        });
    }

    // mask.png — load, validate dimensions. Used for the RLE fallback
    // path; even when polygon.json is present we still validate the mask
    // matches the source so the dataset is internally consistent.
    let mask_path = label_dir.join("mask.png");
    if !mask_path.exists() {
        return Err(LabelError::MissingArtifact {
            label: id,
            artifact: "mask.png",
        });
    }
    let mask_img = image::open(&mask_path)?.to_luma8();
    let mw = mask_img.width();
    let mh = mask_img.height();
    if mw != meta.source_image_width || mh != meta.source_image_height {
        return Err(LabelError::MaskFormat(format!(
            "mask {}x{} does not match source {}x{} in {}",
            mw, mh, meta.source_image_width, meta.source_image_height, id,
        )));
    }

    // Polygon segmentation when polygon.json is present; otherwise RLE.
    let polygon_path = label_dir.join("polygon.json");
    let (bbox, area, segmentation) = if polygon_path.exists() {
        let s = fs::read_to_string(&polygon_path)?;
        let polygon: PolygonJson = serde_json::from_str(&s)?;
        let flat: Vec<f64> = polygon
            .vertices
            .iter()
            .flat_map(|v| [v[0] as f64, v[1] as f64])
            .collect();
        let (bbox, area) = polygon_bbox_and_area(&polygon.vertices);
        (bbox, area, CocoSegmentation::Polygon(vec![flat]))
    } else {
        let raw = mask_img.into_raw();
        let (bbox, area) = mask_bbox_and_area(&raw, mh as usize, mw as usize);
        let counts = encode_rle(&raw, mh as usize, mw as usize);
        (
            bbox,
            area,
            CocoSegmentation::UncompressedRle {
                counts,
                size: [mh, mw],
            },
        )
    };

    let note = build_note(&meta);
    let file_name = format!("{id}/image.png");

    let image = CocoImage {
        id: image_id,
        width: meta.source_image_width,
        height: meta.source_image_height,
        file_name,
        date_captured: meta.created_at.clone(),
        note,
        snapseg_provenance: meta,
    };
    let annotation = CocoAnnotation {
        id: annotation_id,
        image_id,
        category_id: 1,
        bbox,
        area,
        segmentation,
        iscrowd: 0,
    };
    Ok((image, annotation))
}

/// Compute `[x, y, w, h]` bbox in float image-space and shoelace area
/// from a polygon's vertices. The polygon is treated as implicitly
/// closed: the last vertex pairs with the first.
///
/// Returns `([0.0, 0.0, 0.0, 0.0], 0.0)` if the polygon has fewer than
/// three vertices (degenerate; no area).
fn polygon_bbox_and_area(vertices: &[[f32; 2]]) -> ([f64; 4], f64) {
    if vertices.len() < 3 {
        return ([0.0, 0.0, 0.0, 0.0], 0.0);
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for v in vertices {
        let x = v[0] as f64;
        let y = v[1] as f64;
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let n = vertices.len();
    let mut sum: f64 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let xi = vertices[i][0] as f64;
        let yi = vertices[i][1] as f64;
        let xj = vertices[j][0] as f64;
        let yj = vertices[j][1] as f64;
        sum += xi * yj - xj * yi;
    }
    let area = 0.5 * sum.abs();
    ([min_x, min_y, max_x - min_x, max_y - min_y], area)
}

/// Encode a row-major u8 mask as uncompressed COCO RLE
/// (column-major, alternating runs starting from background).
///
/// Pixels above 127 are foreground; everything else is background.
/// If the very first pixel (col=0, row=0) is foreground, the first
/// emitted run-length is a leading zero so the alternation invariant
/// (run #0 = background, run #1 = foreground, …) holds.
fn encode_rle(mask: &[u8], h: usize, w: usize) -> Vec<u32> {
    let mut counts: Vec<u32> = Vec::new();
    let mut current_val: u8 = 0;
    let mut run: u32 = 0;
    for col in 0..w {
        for row in 0..h {
            let v: u8 = if mask[row * w + col] > 127 { 1 } else { 0 };
            if v == current_val {
                run += 1;
            } else {
                counts.push(run);
                current_val = 1 - current_val;
                run = 1;
            }
        }
    }
    counts.push(run);
    counts
}

/// Compute `[x, y, w, h]` bounding box (f64) and foreground pixel
/// count for a row-major u8 mask. Returns `([0,0,0,0], 0.0)` when the
/// mask is empty.
fn mask_bbox_and_area(mask: &[u8], h: usize, w: usize) -> ([f64; 4], f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut area: u64 = 0;
    for row in 0..h {
        for col in 0..w {
            if mask[row * w + col] > 127 {
                area += 1;
                let x = col as f64;
                let y = row as f64;
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if area == 0 {
        return ([0.0, 0.0, 0.0, 0.0], 0.0);
    }
    (
        [min_x, min_y, max_x - min_x + 1.0, max_y - min_y + 1.0],
        area as f64,
    )
}

/// Build the human-readable `note` field:
/// `"<registry_name> · <quality> · <commit-or-dash>"`.
fn build_note(meta: &Provenance) -> String {
    let quality = match meta.operator.quality {
        LabelQuality::Good => "good",
        LabelQuality::NeedsReview => "needs_review",
        LabelQuality::Reject => "reject",
    };
    let commit = meta.runtime.snapseg_commit.as_deref().unwrap_or("-");
    format!("{} · {} · {}", meta.model.registry_name, quality, commit)
}

/// Extract the label id (directory's bottommost component) as a
/// `String`. Falls back to an empty string if the path has no file
/// name component (which the filesystem APIs should never hand us in
/// practice, but we don't `unwrap` in library code).
fn label_id_from_dir(label_dir: &Path) -> String {
    label_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array2;
    use snapseg_core::{GrayImage, Point2, Polarity, Prompt, PromptSession};

    use crate::{
        LabelDir, ModelProvenance, OperatorNote, PolygonJson, ProvenanceInputs, RuntimeProvenance,
        prompts_to_json,
    };

    fn make_gray_4x4() -> GrayImage {
        let data = Array2::from_shape_vec((4, 4), (0u8..16).collect()).expect("4x4 shape");
        GrayImage::from_array(data)
    }

    fn make_mask_4x4() -> Array2<bool> {
        let mut m = Array2::<bool>::default((4, 4));
        m[(1, 1)] = true;
        m[(1, 2)] = true;
        m[(2, 1)] = true;
        m[(2, 2)] = true;
        m
    }

    fn make_inputs(quality: LabelQuality) -> ProvenanceInputs {
        ProvenanceInputs {
            model: ModelProvenance {
                registry_name: "test-model".to_string(),
                family: "test".to_string(),
                encoder_sha256: None,
                decoder_sha256: None,
                model_sha256: Some("deadbeef".to_string()),
            },
            runtime: RuntimeProvenance {
                execution_provider: "CPU".to_string(),
                encoder_ms: None,
                decoder_ms: 7,
                snapseg_commit: Some("abc1234".to_string()),
            },
            operator: OperatorNote {
                note: String::new(),
                quality,
            },
        }
    }

    fn one_click() -> (PromptSession, Vec<u64>) {
        let mut s = PromptSession::new();
        s.push(Prompt::Click {
            point: Point2::new(1.5, 1.5),
            polarity: Polarity::Positive,
        });
        (s, vec![0])
    }

    /// Decode an uncompressed COCO RLE back to a row-major u8 mask
    /// (0 = background, 1 = foreground). Mirror of [`encode_rle`].
    fn decode_rle(counts: &[u32], h: u32, w: u32) -> Vec<u8> {
        let h = h as usize;
        let w = w as usize;
        let mut out = vec![0u8; h * w];
        let mut current_val: u8 = 0;
        let mut linear: usize = 0;
        for &run in counts {
            for _ in 0..run {
                let col = linear / h;
                let row = linear % h;
                out[row * w + col] = current_val;
                linear += 1;
            }
            current_val = 1 - current_val;
        }
        out
    }

    #[test]
    fn convert_single_label_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        let mask = make_mask_4x4();
        let (session, offsets) = one_click();
        let prompts = prompts_to_json(&session, &offsets).expect("prompts_to_json");
        let label = dir
            .save(
                None,
                &gray,
                prompts,
                &mask,
                None,
                None,
                make_inputs(LabelQuality::Good),
            )
            .expect("save");

        // No polygon.json should exist when polygon arg was None; this
        // makes the RLE-fallback precondition visible in the test.
        assert!(
            !label.dir.join("polygon.json").exists(),
            "polygon.json should not exist when polygon arg is None"
        );

        let report = convert_dir(tmp.path(), ConvertOptions::default()).expect("convert_dir");
        let ds = &report.dataset;
        assert_eq!(report.skipped, 0);

        assert_eq!(ds.images.len(), 1);
        assert_eq!(ds.annotations.len(), 1);
        assert_eq!(ds.categories.len(), 1);
        assert_eq!(ds.categories[0].id, 1);
        assert_eq!(ds.categories[0].name, "foreground");

        let img = &ds.images[0];
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.file_name, format!("{}/image.png", label.id));
        assert!(img.note.contains("test-model"));
        assert!(img.note.contains("good"));
        assert!(img.note.contains("abc1234"));

        let ann = &ds.annotations[0];
        assert_eq!(ann.category_id, 1);
        assert_eq!(ann.iscrowd, 0);
        assert!((ann.area - 4.0).abs() < 1e-9);
        // bbox = [min_x, min_y, w, h] over (col=1..=2, row=1..=2).
        let expected_bbox = [1.0_f64, 1.0, 2.0, 2.0];
        for (got, want) in ann.bbox.iter().zip(expected_bbox.iter()) {
            assert!(
                (got - want).abs() < 1e-9,
                "bbox mismatch: got {:?}, want {:?}",
                ann.bbox,
                expected_bbox,
            );
        }

        // RLE roundtrip: decode back to a 4x4 mask and compare.
        let (counts, size) = match &ann.segmentation {
            CocoSegmentation::UncompressedRle { counts, size } => (counts, size),
            other => panic!("expected UncompressedRle, got {other:?}"),
        };
        assert_eq!(*size, [4, 4]);
        // Sanity: counts sum to total pixels.
        let total: u32 = counts.iter().copied().sum();
        assert_eq!(total, 16);
        let decoded = decode_rle(counts, size[0], size[1]);
        for ((row, col), &m) in mask.indexed_iter() {
            let v = decoded[row * 4 + col];
            let want = u8::from(m);
            assert_eq!(
                v, want,
                "RLE decode mismatch at ({row},{col}): got {v}, want {want}",
            );
        }
    }

    #[test]
    fn convert_skips_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        let mask = make_mask_4x4();

        // First label — good.
        let (s1, o1) = one_click();
        let p1 = prompts_to_json(&s1, &o1).expect("prompts");
        dir.save(
            None,
            &gray,
            p1,
            &mask,
            None,
            None,
            make_inputs(LabelQuality::Good),
        )
        .expect("save good");

        // Second label — rejected. Use a different mask so the source-
        // image SHA-256 lands in a different bucket if it depends on
        // mask bytes; the image bytes are unchanged so we expect the
        // region-suffix path (-r2) to be picked.
        let (s2, o2) = one_click();
        let p2 = prompts_to_json(&s2, &o2).expect("prompts");
        dir.save(
            None,
            &gray,
            p2,
            &mask,
            None,
            None,
            make_inputs(LabelQuality::Reject),
        )
        .expect("save rejected");

        // Confirm two label directories exist on disk.
        let label_dirs: Vec<_> = fs::read_dir(tmp.path())
            .expect("readdir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        assert_eq!(label_dirs.len(), 2, "expected two label dirs on disk");

        let with_skip = convert_dir(
            tmp.path(),
            ConvertOptions {
                skip_rejected: true,
                strict: false,
            },
        )
        .expect("convert with skip");
        assert_eq!(with_skip.dataset.images.len(), 1);
        assert_eq!(with_skip.dataset.annotations.len(), 1);
        assert_eq!(with_skip.skipped, 0);

        let no_skip = convert_dir(
            tmp.path(),
            ConvertOptions {
                skip_rejected: false,
                strict: false,
            },
        )
        .expect("convert without skip");
        assert_eq!(no_skip.dataset.images.len(), 2);
        assert_eq!(no_skip.dataset.annotations.len(), 2);
        assert_eq!(no_skip.skipped, 0);
        // Both ids must be present in the no-skip output.
        let names: Vec<&str> = no_skip
            .dataset
            .images
            .iter()
            .map(|i| i.file_name.as_str())
            .collect();
        assert!(names.iter().all(|n| n.ends_with("/image.png")));
    }

    #[test]
    fn convert_skips_partial_dirs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        let mask = make_mask_4x4();
        let (s, o) = one_click();
        let p = prompts_to_json(&s, &o).expect("prompts");
        dir.save(
            None,
            &gray,
            p,
            &mask,
            None,
            None,
            make_inputs(LabelQuality::Good),
        )
        .expect("save");

        // Orphan .partial/ from a hypothetical crashed write.
        fs::create_dir_all(tmp.path().join("orphan.partial")).expect("mkdir partial");

        let report = convert_dir(tmp.path(), ConvertOptions::default()).expect("convert_dir");
        assert_eq!(report.dataset.images.len(), 1);
        assert_eq!(report.dataset.annotations.len(), 1);
        assert_eq!(report.skipped, 0);
    }

    #[test]
    fn convert_uses_polygon_when_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        // 16x16 image (4x4 isn't big enough to embed a (2.5,2.5)..(12,12)
        // triangle).
        let data =
            Array2::from_shape_vec((16, 16), (0u8..=255).take(256).collect()).expect("16x16");
        let gray = GrayImage::from_array(data);
        let mask = Array2::<bool>::default((16, 16));
        let (session, offsets) = one_click();
        let prompts = prompts_to_json(&session, &offsets).expect("prompts_to_json");

        let polygon = PolygonJson {
            schema_version: SCHEMA_VERSION.to_string(),
            vertices: vec![[2.5, 2.5], [10.5, 4.0], [5.0, 12.0]],
            confidence: vec![0.9, 0.95, 0.8],
        };

        dir.save(
            None,
            &gray,
            prompts,
            &mask,
            None,
            Some(&polygon),
            make_inputs(LabelQuality::Good),
        )
        .expect("save");

        let report = convert_dir(tmp.path(), ConvertOptions::default()).expect("convert_dir");
        let ds = &report.dataset;
        assert_eq!(report.skipped, 0);
        assert_eq!(ds.annotations.len(), 1);
        let ann = &ds.annotations[0];

        let flat = match &ann.segmentation {
            CocoSegmentation::Polygon(rings) => {
                assert_eq!(rings.len(), 1, "snapseg emits one polygon per annotation");
                rings[0].clone()
            }
            other => panic!("expected Polygon, got {other:?}"),
        };
        let want_flat = [2.5_f64, 2.5, 10.5, 4.0, 5.0, 12.0];
        assert_eq!(flat.len(), want_flat.len());
        for (got, want) in flat.iter().zip(want_flat.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "polygon flat mismatch: got {flat:?}, want {want_flat:?}"
            );
        }

        // bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        //      = [2.5, 2.5, 8.0, 9.5]
        let want_bbox = [2.5_f64, 2.5, 8.0, 9.5];
        for (got, want) in ann.bbox.iter().zip(want_bbox.iter()) {
            assert!(
                (got - want).abs() < 0.01,
                "bbox mismatch: got {:?}, want {:?}",
                ann.bbox,
                want_bbox
            );
        }

        // Shoelace area:
        // 0.5 * |2.5*4.0 - 10.5*2.5
        //      + 10.5*12.0 - 5.0*4.0
        //      + 5.0*2.5 - 2.5*12.0|
        // = 0.5 * |72.25| = 36.125
        assert!(
            (ann.area - 36.125).abs() < 0.01,
            "area mismatch: got {}",
            ann.area
        );
    }

    #[test]
    fn polygon_segmentation_roundtrips_as_polygon() {
        // Guard the serde(untagged) variant ordering: a JSON array of
        // arrays must deserialize back as Polygon, not get misclassified.
        let seg = CocoSegmentation::Polygon(vec![vec![0.0_f64, 0.0, 1.0, 0.0, 0.5, 1.0]]);
        let json = serde_json::to_string(&seg).expect("serialize");
        let back: CocoSegmentation = serde_json::from_str(&json).expect("deserialize");
        match back {
            CocoSegmentation::Polygon(rings) => {
                assert_eq!(rings.len(), 1);
                assert_eq!(rings[0].len(), 6);
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
    }

    #[test]
    fn encode_rle_leading_zero_when_first_pixel_foreground() {
        // 2x2 mask with (0,0) foreground only. Column-major walk:
        // col0 row0 -> fg (push leading 0, then run=1), col0 row1 -> bg
        // (push 1, run=1), col1 row0 -> bg (run=2), col1 row1 -> bg
        // (run=3). Final push 3 → [0, 1, 3].
        let mask = vec![
            255, 0, // row 0
            0, 0, // row 1
        ];
        let counts = encode_rle(&mask, 2, 2);
        assert_eq!(counts, vec![0, 1, 3], "got {counts:?}");
    }

    #[test]
    fn convert_skips_corrupt_label_in_default_mode() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        // Save one good label.
        let gray = make_gray_4x4();
        let mask = make_mask_4x4();
        let (session, offsets) = one_click();
        let prompts = prompts_to_json(&session, &offsets).expect("prompts_to_json");
        dir.save(
            None,
            &gray,
            prompts,
            &mask,
            None,
            None,
            make_inputs(LabelQuality::Good),
        )
        .expect("save good");

        // Create a sibling directory with a corrupt meta.toml so that
        // the sorted traversal encounters it (name "corrupt" sorts before
        // typical sha8 names).
        let corrupt_dir = tmp.path().join("0corrupt");
        fs::create_dir_all(&corrupt_dir).expect("mkdir corrupt");
        fs::write(corrupt_dir.join("meta.toml"), b"this is not toml!@#$").expect("write bad toml");
        // Provide the other expected artifacts so only meta.toml is corrupt.
        let small_png: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG sig
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x00, 0x00, 0x00, 0x00, 0x3a, 0x7e, 0x9b,
            0x55, // bit depth=8, color=Gray, crc
            0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, // IDAT length + type
            0x78, 0x9c, 0x62, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, // zlib-compressed 1 pixel
            0xe2, 0x21, 0xbc, 0x33, // IDAT crc
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82, // IEND
        ];
        fs::write(corrupt_dir.join("image.png"), small_png).expect("write image.png");
        fs::write(corrupt_dir.join("mask.png"), small_png).expect("write mask.png");
        fs::write(corrupt_dir.join("prompts.json"), b"[]").expect("write prompts.json");

        // Lenient mode (default): returns Ok with the one good image, skipped == 1.
        let report = convert_dir(tmp.path(), ConvertOptions::default())
            .expect("lenient convert should succeed");
        assert_eq!(
            report.dataset.images.len(),
            1,
            "expected one good image, got {:?}",
            report
                .dataset
                .images
                .iter()
                .map(|i| &i.file_name)
                .collect::<Vec<_>>()
        );
        assert_eq!(report.skipped, 1, "expected 1 skipped label");

        // Strict mode: returns Err on the corrupt label.
        let strict_result = convert_dir(
            tmp.path(),
            ConvertOptions {
                strict: true,
                ..ConvertOptions::default()
            },
        );
        assert!(
            strict_result.is_err(),
            "strict mode should abort on corrupt label"
        );
    }
}
