use std::sync::{Arc, Mutex};
use std::net::{TcpListener, TcpStream};
use std::io::Read;
use std::thread;

struct ShortMem {
    last_user: Option<String>,
    last_assistant: Option<String>,
}


fn main() {
    let listener = TcpListener::bind("127.0.0.1:9090").unwrap();
    
}
