//! `snapseg` — interactive deep-segmentation desktop app.
//!
//! This iteration is **UI without a model**: the user can open a grayscale
//! image, toggle positive/negative click tools, place clicks on the canvas
//! (primary mouse = current tool, secondary mouse = opposite polarity),
//! and clear them. The accumulated `PromptSession` is what we'll feed to a
//! real segmenter once `ort` + an adapter are wired in the next step.

use std::path::PathBuf;

use anyhow::{Context, Result};
use eframe::egui;
use snapseg_core::{GrayImage, Point2, Polarity, Prompt, PromptSession};

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
    /// Last error to surface in the status bar.
    error: Option<String>,
}

impl Default for SnapsegApp {
    fn default() -> Self {
        Self {
            image: None,
            current_polarity: Polarity::Positive,
            session: PromptSession::new(),
            error: None,
        }
    }
}

struct LoadedImage {
    path: PathBuf,
    gray: GrayImage,
    texture: egui::TextureHandle,
}

impl SnapsegApp {
    fn open_dialog(&mut self, ctx: &egui::Context) {
        let path = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
            .pick_file();
        let Some(path) = path else { return };
        match load_image(&path, ctx) {
            Ok(loaded) => {
                tracing::info!(
                    path = %loaded.path.display(),
                    w = loaded.gray.width,
                    h = loaded.gray.height,
                    "image loaded"
                );
                self.image = Some(loaded);
                self.session.clear();
                self.error = None;
            }
            Err(e) => {
                tracing::error!("failed to load image: {e:#}");
                self.error = Some(format!("Load failed: {e}"));
            }
        }
    }

    fn draw_canvas(&mut self, ui: &mut egui::Ui) {
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
                    self.open_dialog(ctx);
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
                }

                ui.add_space(8.0);
                ui.separator();
                ui.label("Model: (none loaded)");
                ui.label("EP: (n/a)");
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
                self.draw_canvas(ui);
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

/// Largest rect with the same aspect as `inner_size` that fits in
/// `available`, centered within it, anchored at `origin`.
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
