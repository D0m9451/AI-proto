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

    format!("Context:\n{}\n{}", user, model)
}


fn guilisten<T>() {
    let listener = TcpListener::bind("127.0.0:1:9090").unwrap();
    for stream in listener.incoming() {
        let user = Arc::new(Mutex::new(None::<T>));
    }
}

fn modellisten<T>() {
    let listener = TcpListener::bind("127.0.0:1:8080").unwrap();
    for stream in listener.incoming() {
        let model = Arc::new(Mutex::new(None::<T>));
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




fn main() {
    //guilisten();

    let mem = ShortMem {
        user: Some("Hello".to_string()),
        model: Some("Hi!".to_string()),
    };


    let prompt = construct(&mem);
    println!("{}", prompt); 
}
