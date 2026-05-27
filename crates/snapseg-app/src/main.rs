//! `snapseg` — interactive deep-segmentation desktop app.
//!
//! This iteration wires the full loop end-to-end against a MobileSAM ONNX
//! pair (encoder + decoder). The user:
//!   1. opens a grayscale image,
//!   2. picks the two ONNX files via "Load MobileSAM…",
//!   3. places positive / negative clicks (primary mouse = current
//!      polarity; secondary mouse = opposite),
//!   4. watches the predicted mask update after each click.
//!
//! Inference runs synchronously on the UI thread — the first encoder
//! pass takes ~1–2s on CPU; clicks after that hit only the decoder.
//! Off-thread inference + cancellation belongs in the next iteration.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use eframe::egui;
use ndarray::Array2;
use snapseg_core::{GrayImage, InteractiveSegmenter, Point2, Polarity, Prompt, PromptSession};
use snapseg_models::mobile_sam::MobileSamSegmenter;
use snapseg_runtime::RuntimeConfig;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("snapseg"),
        ..Default::default()
    };

    eframe::run_native(
        "snapseg",
        options,
        Box::new(|_cc| Ok(Box::new(SnapsegApp::default()))),
    )
}

struct SnapsegApp {
    image: Option<LoadedImage>,
    /// Polarity attached to a primary-button click; secondary-button click
    /// uses the opposite. Keeps the workflow single-handed.
    current_polarity: Polarity,
    session: PromptSession,
    error: Option<String>,
    segmenter: Option<Box<dyn InteractiveSegmenter>>,
    segmenter_label: Option<String>,
    /// Embedding state: Some(true) means `set_image` has run on the
    /// current image successfully and clicks can be processed.
    embedding_ready: bool,
    mask_texture: Option<egui::TextureHandle>,
    last_inference_ms: Option<u64>,
}

impl Default for SnapsegApp {
    fn default() -> Self {
        Self {
            image: None,
            current_polarity: Polarity::Positive,
            session: PromptSession::new(),
            error: None,
            segmenter: None,
            segmenter_label: None,
            embedding_ready: false,
            mask_texture: None,
            last_inference_ms: None,
        }
    }
}

struct LoadedImage {
    path: PathBuf,
    gray: GrayImage,
    texture: egui::TextureHandle,
}

impl SnapsegApp {
    fn open_image_dialog(&mut self, ctx: &egui::Context) {
        let path = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
            .pick_file();
        let Some(path) = path else { return };
        match load_image(&path, ctx) {
            Ok(loaded) => {
                tracing::info!(
                    path = %loaded.path.display(),
                    w = loaded.gray.width, h = loaded.gray.height,
                    "image loaded"
                );
                self.image = Some(loaded);
                self.session.clear();
                self.mask_texture = None;
                self.last_inference_ms = None;
                self.embedding_ready = false;
                self.error = None;
                self.run_set_image();
            }
            Err(e) => {
                tracing::error!("failed to load image: {e:#}");
                self.error = Some(format!("Load failed: {e}"));
            }
        }
    }

    fn load_mobile_sam_dialog(&mut self) {
        let enc = rfd::FileDialog::new()
            .add_filter("ONNX", &["onnx"])
            .set_title("Select MobileSAM encoder.onnx")
            .pick_file();
        let Some(enc) = enc else { return };
        let dec = rfd::FileDialog::new()
            .add_filter("ONNX", &["onnx"])
            .set_title("Select MobileSAM decoder.onnx")
            .pick_file();
        let Some(dec) = dec else { return };

        let mut parts = std::collections::HashMap::new();
        parts.insert("encoder".to_string(), enc);
        parts.insert("decoder".to_string(), dec);
        let config = RuntimeConfig::default();
        match MobileSamSegmenter::from_parts(
            "mobile-sam".to_string(),
            &parts,
            1024,
            &config,
        ) {
            Ok(seg) => {
                tracing::info!("MobileSAM loaded");
                self.segmenter_label = Some(format!(
                    "{} ({})",
                    seg.name(),
                    "CPU" // EP picker comes next; for now CPU only.
                ));
                self.segmenter = Some(Box::new(seg));
                self.embedding_ready = false;
                self.error = None;
                self.run_set_image();
            }
            Err(e) => {
                tracing::error!("MobileSAM load failed: {e}");
                self.error = Some(format!("MobileSAM load: {e}"));
            }
        }
    }

    fn run_set_image(&mut self) {
        let (Some(img), Some(seg)) = (&self.image, self.segmenter.as_mut()) else {
            return;
        };
        let started = Instant::now();
        match seg.set_image(&img.gray) {
            Ok(()) => {
                self.embedding_ready = true;
                tracing::info!(
                    ms = started.elapsed().as_millis() as u64,
                    "encoder pass complete"
                );
            }
            Err(e) => {
                tracing::error!("set_image failed: {e}");
                self.error = Some(format!("set_image: {e}"));
                self.embedding_ready = false;
            }
        }
    }

    fn run_segment(&mut self, ctx: &egui::Context) {
        if !self.embedding_ready {
            return;
        }
        let Some(seg) = self.segmenter.as_mut() else { return };
        if self.session.is_empty() {
            self.mask_texture = None;
            self.last_inference_ms = None;
            return;
        }
        match seg.segment(&self.session) {
            Ok(res) => {
                self.last_inference_ms = Some(res.inference_time.as_millis() as u64);
                self.mask_texture = Some(mask_to_texture(ctx, &res.mask));
                self.error = None;
            }
            Err(e) => {
                tracing::error!("segment failed: {e}");
                self.error = Some(format!("segment: {e}"));
            }
        }
    }

    fn draw_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(img) = &self.image else { return };

        let avail = ui.available_size();
        let img_size = egui::vec2(img.gray.width as f32, img.gray.height as f32);
        let display_rect = fit_rect(img_size, avail, ui.cursor().min);

        let response = ui.allocate_rect(display_rect, egui::Sense::click());
        let painter = ui.painter_at(display_rect);

        painter.image(
            img.texture.id(),
            display_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // Mask overlay
        if let Some(mask_tex) = &self.mask_texture {
            painter.image(
                mask_tex.id(),
                display_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        let mut clicked = false;
        if response.clicked() || response.secondary_clicked() {
            let secondary = response.secondary_clicked();
            if let Some(pos) = response.interact_pointer_pos() {
                let img_pt = screen_to_image(pos, display_rect, img_size);
                let polarity = if secondary {
                    opposite(self.current_polarity)
                } else {
                    self.current_polarity
                };
                self.session.push(Prompt::Click {
                    point: img_pt,
                    polarity,
                });
                clicked = true;
            }
        }

        for prompt in &self.session.prompts {
            if let Prompt::Click { point, polarity } = prompt {
                let screen = image_to_screen(*point, display_rect, img_size);
                let (fill, stroke) = match polarity {
                    Polarity::Positive => (
                        egui::Color32::from_rgb(80, 220, 100),
                        egui::Color32::BLACK,
                    ),
                    Polarity::Negative => (
                        egui::Color32::from_rgb(230, 90, 90),
                        egui::Color32::BLACK,
                    ),
                };
                painter.circle_filled(screen, 6.0, fill);
                painter.circle_stroke(screen, 6.0, egui::Stroke::new(1.5, stroke));
            }
        }

        if clicked {
            self.run_segment(ctx);
        }
    }
}

impl eframe::App for SnapsegApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::right("controls")
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("snapseg");
                ui.separator();

                if ui.button("Open image…").clicked() {
                    self.open_image_dialog(ctx);
                }
                if ui.button("Load MobileSAM…").clicked() {
                    self.load_mobile_sam_dialog();
                }

                ui.add_space(8.0);
                ui.label("Click tool:");
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.current_polarity,
                        Polarity::Positive,
                        "● Positive",
                    );
                    ui.selectable_value(
                        &mut self.current_polarity,
                        Polarity::Negative,
                        "✕ Negative",
                    );
                });
                ui.label(
                    egui::RichText::new("Right-click = opposite polarity")
                        .small()
                        .weak(),
                );

                ui.add_space(8.0);
                ui.separator();
                ui.label(format!("Prompts: {}", self.session.prompts.len()));
                if ui.button("Clear prompts").clicked() {
                    self.session.clear();
                    self.mask_texture = None;
                    self.last_inference_ms = None;
                }

                ui.add_space(8.0);
                ui.separator();
                match &self.segmenter_label {
                    Some(s) => ui.label(format!("Model: {s}")),
                    None => ui.label("Model: (none loaded)"),
                };
                ui.label(format!(
                    "Embedding: {}",
                    if self.embedding_ready { "ready" } else { "pending" }
                ));
                if let Some(ms) = self.last_inference_ms {
                    ui.label(format!("Last segment: {ms} ms"));
                }
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::LIGHT_RED, err);
                } else if let Some(img) = &self.image {
                    ui.label(format!(
                        "{}  •  {}×{}  •  prompts: {}",
                        img.path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                        img.gray.width,
                        img.gray.height,
                        self.session.prompts.len(),
                    ));
                } else {
                    ui.label("idle — open an image to begin");
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.image.is_some() {
                self.draw_canvas(ui, ctx);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an image (right panel) to begin.");
                });
            }
        });
    }
}

fn opposite(p: Polarity) -> Polarity {
    match p {
        Polarity::Positive => Polarity::Negative,
        Polarity::Negative => Polarity::Positive,
    }
}

fn fit_rect(inner_size: egui::Vec2, available: egui::Vec2, origin: egui::Pos2) -> egui::Rect {
    let scale = (available.x / inner_size.x).min(available.y / inner_size.y);
    let scaled = inner_size * scale;
    let offset = (available - scaled) * 0.5;
    egui::Rect::from_min_size(origin + offset, scaled)
}

fn screen_to_image(p: egui::Pos2, display_rect: egui::Rect, img_size: egui::Vec2) -> Point2 {
    let u = (p.x - display_rect.min.x) / display_rect.width();
    let v = (p.y - display_rect.min.y) / display_rect.height();
    Point2::new(u * img_size.x, v * img_size.y)
}

fn image_to_screen(pt: Point2, display_rect: egui::Rect, img_size: egui::Vec2) -> egui::Pos2 {
    let u = pt.x / img_size.x;
    let v = pt.y / img_size.y;
    egui::pos2(
        display_rect.min.x + u * display_rect.width(),
        display_rect.min.y + v * display_rect.height(),
    )
}

fn load_image(path: &PathBuf, ctx: &egui::Context) -> Result<LoadedImage> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();

    let gray = GrayImage::from_array(
        ndarray::Array2::from_shape_vec((h, w), raw.clone()).context("ndarray shape mismatch")?,
    );

    let mut pixels = Vec::with_capacity(w * h);
    for &v in &raw {
        pixels.push(egui::Color32::from_gray(v));
    }
    let color_image = egui::ColorImage {
        size: [w, h],
        pixels,
    };
    let texture = ctx.load_texture(
        path.file_name().and_then(|s| s.to_str()).unwrap_or("image"),
        color_image,
        egui::TextureOptions::LINEAR,
    );

    Ok(LoadedImage {
        path: path.clone(),
        gray,
        texture,
    })
}

fn mask_to_texture(ctx: &egui::Context, mask: &Array2<bool>) -> egui::TextureHandle {
    let (h, w) = mask.dim();
    let mut pixels = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            if mask[(y, x)] {
                pixels.push(egui::Color32::from_rgba_premultiplied(40, 110, 200, 110));
            } else {
                pixels.push(egui::Color32::TRANSPARENT);
            }
        }
    }
    let img = egui::ColorImage {
        size: [w, h],
        pixels,
    };
    ctx.load_texture("mask", img, egui::TextureOptions::NEAREST)
}
