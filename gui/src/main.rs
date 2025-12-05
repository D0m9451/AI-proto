#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::{egui, App, Frame, NativeOptions};
use std::sync::{Arc, Mutex};
use std::net::{TcpListener, TcpStream};
use std::io::Read;
use std::thread;
#[derive(PartialEq)]





enum AppState {
    MainMenu,
    Chat,
    Settings,
    ModelInfo,
}
enum Theme {
    Light,
    Dark,
    Blue,
    Forest,
    Cherry,
    Greyscale,
    Punk,
    DarkPunk,
    Infernus,

}

struct VinnyApp {
    state: AppState,
    theme: Theme,
    max: i32,
    temp:f32,
    topP: f32,
    repPen: f32,
    input: String,
    messages: Vec<String>,
    receiver: Arc<Mutex<String>>,
    connect: Arc<Mutex<String>>,
    listener: Option<std::thread::JoinHandle<()>>,
}

impl Default for VinnyApp {
    fn default() -> Self {
        let receiver = Arc::new(Mutex::new(String::new()));
        let connect = Arc::new(Mutex::new(String::from("Ready")));
        let text_clone = Arc::clone(&receiver);
        let status_clone = Arc::clone(&connect);
        
        // Start listener thread
        let handle = thread::spawn(move || {
            let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
            println!("✓ Listening on port 8080");
            
            for stream in listener.incoming() {
                if let Ok(stream) = stream {
                    *status_clone.lock().unwrap() = String::from("🟢 Receiving...");
                    
                    let mut buffer = [0; 1024];
                    let mut stream = stream;
                    loop {
                        match stream.read(&mut buffer) {
                            Ok(0) => break,
                            Ok(n) => {
                                let chunk = String::from_utf8_lossy(&buffer[..n]);
                                text_clone.lock().unwrap().push_str(&chunk);
                            }
                            Err(_) => break,
                        }
                    }
                    
                    *status_clone.lock().unwrap() = String::from("Ready");
                }
            }
        });

        Self {
            state: AppState::MainMenu,
            theme: Theme::Dark,
            max: 200,
            temp: 0.7,
            topP: 0.85,
            repPen: 1.2,
            input: String::new(),
            messages: vec![],
            receiver,
            connect,
            listener: Some(handle),
        }
    }
}

impl App for VinnyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {

        match self.theme {
            Theme::Light => ctx.set_visuals(egui::Visuals::light()),

            Theme::Dark => ctx.set_visuals(egui::Visuals::dark()),

            Theme::Blue => {
                let mut visuals = egui::Visuals::dark();
                visuals.window_fill = egui::Color32::from_rgb(10, 25, 50);          // Color 1
                visuals.panel_fill = egui::Color32::from_rgb(20, 50, 90);           // Color 2
                visuals.override_text_color = Some(egui::Color32::from_rgb(240, 245, 250)); // Color 6
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(255, 140, 90);     // Color 4
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(255, 180, 120);   // Color 5
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(40, 80, 130);    // Color 3
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(20, 35, 60); // Color 10
                ctx.set_visuals(visuals);
            }

            Theme::Forest => {
                let mut visuals = egui::Visuals::dark();
                visuals.window_fill = egui::Color32::from_rgb(34, 49, 32);          // Color 1
                visuals.panel_fill = egui::Color32::from_rgb(50, 70, 45);           // Color 2
                visuals.override_text_color = Some(egui::Color32::from_rgb(201, 184, 143)); // Color 6
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(112, 142, 90);     // Color 4
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(153, 125, 90);    // Color 5
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(71, 101, 57);    // Color 3
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(45, 60, 40); // Color 10
                ctx.set_visuals(visuals);

            }

            Theme::Cherry => {
                let mut visuals = egui::Visuals::light();
                visuals.window_fill = egui::Color32::from_rgb(255, 240, 245);         // Color 1
                visuals.panel_fill = egui::Color32::from_rgb(255, 210, 220);          // Color 2
                visuals.override_text_color = Some(egui::Color32::from_rgb(231, 84, 128)); // Color 8
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(255, 120, 150);     // Color 4
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(255, 160, 180);    // Color 5
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(255, 180, 200);   // Color 3
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(150, 100, 110); // Color 10
                ctx.set_visuals(visuals);


            }

            Theme::Greyscale => {
                let mut visuals = egui::Visuals::light();
                visuals.window_fill = egui::Color32::from_rgb(200, 200, 200);        // Color 1
                visuals.panel_fill = egui::Color32::from_rgb(180, 180, 180);         // Color 2
                visuals.override_text_color = Some(egui::Color32::from_rgb(20, 20, 20)); // Color 10
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(100, 100, 100); // Color 6
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(120, 120, 120); // Color 5
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(160, 160, 160); // Color 3
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(140, 140, 140); // Color 4
                ctx.set_visuals(visuals);



            }
            Theme::Punk => {
                let mut visuals = egui::Visuals::dark();
                visuals.window_fill = egui::Color32::from_rgb(0, 255, 13);               
                visuals.panel_fill = egui::Color32::from_rgb(125, 36, 163);  
                visuals.override_text_color = Some(egui::Color32::from_rgb(0, 255, 13));
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                ctx.set_visuals(visuals);
            }
            Theme::Infernus => {
                let mut visuals = egui::Visuals::dark();
                visuals.window_fill = egui::Color32::from_rgb(15, 0, 0);
                visuals.panel_fill = egui::Color32::from_rgb(40, 5, 5);
                visuals.override_text_color = Some(egui::Color32::from_rgb(255, 220, 100));
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(90, 10, 10);
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(150, 20, 20);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(210, 30, 30);
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(255, 60, 20);
                ctx.set_visuals(visuals);

            }
            Theme::DarkPunk => {
                let mut visuals = egui::Visuals::dark();
                visuals.window_fill = egui::Color32::from_rgb(5, 8, 41);
                visuals.panel_fill = egui::Color32::from_rgb(5, 8, 41);
                visuals.override_text_color = Some(egui::Color32::from_rgb(23, 232, 0));
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(255, 0, 208);
                ctx.set_visuals(visuals);
            }
        }

        match self.state {
            AppState::MainMenu => self.show_main_menu(ctx),
            AppState::Chat => self.show_chat(ctx),
            AppState::Settings => self.show_settings(ctx),
            AppState::ModelInfo => self.show_model_info(ctx),

        }
    }
}

impl VinnyApp {
    fn show_main_menu(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Vinny the AI assistant");
                ui.add_space(30.0); 

                if ui.button("💬 Start chat").clicked() {
                    self.state = AppState::Chat;
                }
                ui.add_space(10.0);

                if ui.button("⚙ Settings").clicked() {
                    self.state = AppState::Settings;
                }
                ui.add_space(10.0);

                if ui.button("📝 Model Info").clicked() {
                    self.state = AppState::ModelInfo;
                }
                ui.add_space(10.0);

                if ui.button("🚪Exit").clicked() {
                    std::process::exit(0);
                }
            });
        });
    }


    fn show_chat(&mut self, ctx: &egui::Context) {
        ctx.request_repaint();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("💬 Chat room");
            ui.label("Chat room: ");

            egui::SidePanel::right("right_panel")
                .default_width(220.0)
                .resizable(true)
                .show(ctx, |ui| {
                    // Reserve space for input box at bottom
                    let input_height = 20.0; // Adjust based on your needs
                    let available_height = ui.available_height();
                    let messages_height = available_height - input_height;

                    // Messages scroll area
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(messages_height)
                        .show(ui, |ui| {
                            // Show previous messages
                            for msg in &self.messages {
                                ui.group(|ui| {
                                    ui.label(format!("User: {}", msg));
                                });
                            }
        
                            // Show streaming AI response
                            let ai_text = self.receiver.lock().unwrap();
                            if !ai_text.is_empty() {
                                ui.group(|ui| {
                                    ui.label(format!("Vinny: {}", &*ai_text));
                                });
                            }
                        });

                    ui.separator();

                    // Input box (always visible at bottom)
                    ui.horizontal(|ui| {
                        let input = ui.text_edit_multiline(&mut self.input);

                        if ui.button("Send ➡️").clicked() {
                            self.send();
                        }

                        if input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            self.send();
                        }
                    });
                });
    

            
            egui::SidePanel::left("left_panel")
                .default_width(220.0)   // width of the left panel
                .resizable(true)        // allow user to resize (optional)
                .show(ctx, |ui| {
                    ui.heading("Model Settings");
                    ui.separator();
                    ui.label("Max Tokens: ");
                    ui.add(
                        egui::Slider::new(&mut self.max, 0..=1000)
                    );
                    ui.separator();

                    ui.label("Temperature: ");
                    ui.add(
                        egui::Slider::new(&mut self.temp, 0.0..=1.0)
                    );
                    ui.separator();
                    
                    ui.label("Top P: ");
                    ui.add(
                        egui::Slider::new(&mut self.topP, 0.0..=10.0)
                    );
                    ui.separator();

                    ui.label("Repetition penalty: ");
                    ui.add(
                        egui::Slider::new(&mut self.repPen, 0.0..=10.0)
                    );
                    ui.separator();

            });

            ui.add_space(325.0);
            if ui.button("Back to Menu").clicked() {
                self.state = AppState::MainMenu;
            }
        });
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        let mut gpuq = false;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("⚙ Settings");

            ui.horizontal_wrapped(|ui| {
                ui.heading("Theme: ");
                if ui.button("Light").clicked() {
                    self.theme = Theme::Light;
                }
                if ui.button("Dark").clicked() {
                    self.theme = Theme::Dark;
                }
                if ui.button("Blue").clicked() {
                    self.theme = Theme::Blue;
                }
                if ui.button("Forest").clicked() {
                    self.theme = Theme::Forest;
                }
                if ui.button("Cherry").clicked() {
                    self.theme = Theme::Cherry;
                }
                if ui.button("Greyscale").clicked() {
                    self.theme = Theme::Greyscale
                }
                if ui.button("Punk").clicked() {
                    self.theme = Theme::Punk
                }
                if ui.button("Dark Punk").clicked() {
                    self.theme = Theme::DarkPunk
                }
                if ui.button("Infernus").clicked() {
                    self.theme = Theme::Infernus
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.heading("Performance: ");

                if ui.checkbox(&mut gpuq, "GPU quantization:").changed() {
                    println!("Checkbox toggled! New value: {gpuq}");
                }
            });

            if ui.button("Back to Menu").clicked() {
                self.state = AppState::MainMenu;
            }
        });
    }
    fn show_model_info(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📝 Model Info");
            egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.label("This program uses the AI model Qwen2.5-3B.
                Information on this medel is desplayed below:

                    Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. Qwen2.5 brings the following improvements upon Qwen2:

                        Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.
                        Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.
                        Long-context Support up to 128K tokens and can generate up to 8K tokens.
                        Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

                    This program uses the base 3B Qwen2.5 model, which has the following features:

                        Type: Causal Language Models
                        Training Stage: Pretraining
                        Architecture: transformers with RoPE, SwiGLU, RMSNorm, Attention QKV bias and tied word embeddings
                        Number of Parameters: 3.09B
                        Number of Paramaters (Non-Embedding): 2.77B
                        Number of Layers: 36
                        Number of Attention Heads (GQA): 16 for Q and 2 for KV
                        Context Length: Full 32,768 tokens");

                        });

            if ui.button("Back to Menu").clicked() {
                self.state = AppState::MainMenu;
            }
        });
    }

    fn send(&mut self) {
        if !self.input.trim().is_empty() {
            self.messages.push(self.input.clone());
            self.input.clear();
        }
    }

}

fn main() {
    let native_options = NativeOptions::default();
    eframe::run_native(
        "Vinny",
        native_options,
        
        Box::new(|cc|{
            cc.egui_ctx.set_visuals(egui::Visuals::dark());

            Ok(Box::new(VinnyApp::default()))
        }),
    ).unwrap();
}

