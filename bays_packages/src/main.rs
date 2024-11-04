mod modules;
mod simple_mean_online;
mod simple_mean_bulk;

use std::env;

// ---------------------------------------------------------------------
fn main() {
    let args: Vec<String> = env::args().collect();
    // 実行ファイル名を除いて、2つの引数が必要
    if args.len() < 2 {
        eprintln!("Error: Missing arguments.\nUsage: {} <arg>", args[0]);
        std::process::exit(1); // エラー終了コードを返す
    }

    if args[1] == "--simple_mean_online" { //cargo runを使う前提。cargo run -p bays_packages -- --simple_mean_online
        simple_mean_online::run();
    } else if args[1] == "--simple_mean_bulk" {
        simple_mean_bulk::run();
    } else {
        panic!("有効な引数ではありません!");
    }
}

