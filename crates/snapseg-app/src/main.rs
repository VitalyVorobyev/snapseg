//! `snapseg` — interactive deep-segmentation desktop app.
//!
//! Skeleton: opens a window, will host the image canvas + side panel +
//! status bar described in the plan. Loading images, running adapters,
//! and the subpixel-edges toggle are wired up in subsequent iterations.

use eframe::egui;

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
            .with_title("snapseg"),
        ..Default::default()
    };

    eframe::run_native(
        "snapseg",
        options,
        Box::new(|_cc| Ok(Box::new(DeepsegApp::default()))),
    )
}

#[derive(Default)]
struct DeepsegApp {}

impl eframe::App for DeepsegApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::right("controls")
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("snapseg");
                ui.separator();
                ui.label("Model: (none loaded)");
                ui.label("EP: (n/a)");
                ui.separator();
                ui.label("Open an image to begin.");
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("idle");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label("(image canvas placeholder)");
            });
        });
    }
}
