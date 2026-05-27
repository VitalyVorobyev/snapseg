//! `snapseg` — interactive deep-segmentation desktop app.
//!
//! Entry point only. The eframe loop drives [`crate::app::SnapsegApp`];
//! everything else lives in dedicated modules:
//!
//! - [`crate::app`]      — top-level `eframe::App` state + update wiring
//! - [`crate::canvas`]   — image canvas drawing, click capture
//! - [`crate::panel`]    — right-side controls + bottom status bar
//! - [`crate::dialogs`]  — file dialogs for image + model load
//! - [`crate::inference`] — synchronous encoder/decoder calls
//! - [`crate::coords`]   — screen ↔ image coordinate transforms
//! - [`crate::textures`] — egui texture building from grayscale / mask

mod app;
mod canvas;
mod coords;
mod dialogs;
mod inference;
mod panel;
mod textures;

use eframe::egui;

use crate::app::SnapsegApp;

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
