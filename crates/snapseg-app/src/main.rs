//! `snapseg` — interactive deep-segmentation desktop app.
//!
//! Entry point only. The eframe loop drives [`crate::app::SnapsegApp`];
//! everything else lives in dedicated modules:
//!
//! - [`crate::app`]      — top-level `eframe::App` state + update wiring
//! - [`crate::canvas`]   — image canvas drawing, click capture
//! - [`crate::config`]   — project-local `snapseg.toml` operator config
//! - [`crate::panel`]    — right-side controls + bottom status bar
//! - [`crate::dialogs`]  — file dialogs for image + model load
//! - [`crate::inference`] — synchronous encoder/decoder calls
//! - [`crate::coords`]   — screen ↔ image coordinate transforms
//! - [`crate::textures`] — egui texture building from grayscale / mask

mod app;
mod canvas;
mod config;
mod coords;
mod dialogs;
mod inference;
mod labels;
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

    let app = match crate::config::AppConfig::load_default() {
        Ok(Some(cfg)) => {
            tracing::info!("loaded snapseg.toml");
            // why this is unsafe: std::env::set_var is unsafe in the 2024
            // edition because environment access isn't thread-safe. Done
            // here in main, before any ort session or worker thread exists.
            if let Some(p) = cfg.onnxruntime_path.as_ref() {
                tracing::info!(path = %p.display(), "setting ORT_DYLIB_PATH from config");
                unsafe { std::env::set_var("ORT_DYLIB_PATH", p) };
            }
            SnapsegApp::with_config(cfg)
        }
        Ok(None) => {
            tracing::debug!("no snapseg.toml; using built-in defaults");
            SnapsegApp::default()
        }
        Err(e) => {
            tracing::error!("failed to load snapseg.toml: {e}");
            SnapsegApp::default()
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("snapseg"),
        ..Default::default()
    };

    eframe::run_native("snapseg", options, Box::new(|_cc| Ok(Box::new(app))))
}
