use std::sync::{Arc, Mutex};
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;

struct ShortMem {
    user: Option<String>,
    model: Option<String>,
}

fn construct(mem: &ShortMem) -> String {
    let user = mem.user.as_deref().unwrap_or("");
    let model = mem.model.as_deref().unwrap_or("");

    format!("Context:\n User: {}\n Vinny: {}", user, model)
}





fn guilisten(mem: Arc<Mutex<ShortMem>>) {

    let listener = TcpListener::bind("127.0.0:1:9090").unwrap();
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut buffer = String::new();
        stream.read_to_string(&mut buffer).unwrap();

        let mut mem = mem.lock().unwrap();
        mem.user = Some(buffer.trim().to_string());
        mem.model = None;
    }
}

fn modellisten(mem: Arc<Mutex<ShortMem>>) {
    let listener = TcpListener::bind("127.0.0:1:8080").unwrap();

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut buffer = String::new();
        stream.read_to_string(&mut buffer).unwrap();

        let mut mem = mem.lock().unwrap();
        mem.model = Some(buffer.trim().to_string());
    }
}






fn sendgui() {
    let mut stream = TcpStream::connect("127.0.0:1:9090").unwrap();
    stream.write_all(b"RAHGOOO").unwrap();
}

fn sendmodel() {
    let mut stream = TcpStream::connect("127.0.0:1:8080").unwrap();
    stream.write_all(b"GOOORAH").unwrap();
}




fn short() {
    let mem = Arc::new(Mutex::new(ShortMem {
        user: None,
        model: None,
    }));


    let memgui = Arc::clone(&mem);
    let memmodel = Arc::clone(&mem);

    thread::spawn(move || { guilisten(memgui);});
    thread::spawn(move || { modellisten(memmodel);});
    loop {
        let memlock = mem.lock().unwrap();
        let prompt = construct(&memlock);
        drop(memlock);

        println!("{}", prompt);
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

}

/*-------------------------------------------------------------------------------------------------------------------------- */

fn main() {
    short();
}