#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
//#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use eframe::{egui, App, Frame, NativeOptions};

#[derive(PartialEq)]
enum AppState {
    MainMenu,
    Chat,
    Settings,
}

struct VinnyApp {
    state: AppState,
}

impl Default for VinnyApp {
    fn default() -> Self {
        Self {
            state: AppState::MainMenu,
        }
    }
}

impl App for VinnyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        match self.state {
            AppState::MainMenu => self.show_main_menu(ctx),
            AppState::Chat => self.show_chat(ctx),
            AppState::Settings => self.show_settings(ctx),
        }
    }
}

impl VinnyApp {
    fn show_main_menu(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Vinny the AI assistant");
            ui.separator();

            if ui.button("Start Chat").clicked() {
                self.state = AppState::Chat;
            }

            if ui.button("Settings").clicked() {
                self.state = AppState::Settings;
            }

            if ui.button("Exit").clicked() {
                std::process::exit(0);
            }
        });
    }

    fn show_chat(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("helo");
            ui.label("Chat room bit");

            if ui.button("Back to Menu").clicked() {
                self.state = AppState::MainMenu;
            }
        });
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("⚙ Settings");
            ui.label("settings bit");

            if ui.button("Back to Menu").clicked() {
                self.state = AppState::MainMenu;
            }
        });
    }
}

fn main() {
    let native_options = NativeOptions::default();
    eframe::run_native(
        "Vinny",
        native_options,
        Box::new(|_cc| Ok(Box::new(VinnyApp::default()))),
    ).unwrap();
}

