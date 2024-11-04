// mod modules;
// mod calculation_plnans;
mod modules;
mod simple_mean_online;
//use calculation_plnans::simple_mean_online;
use std::env;

// ---------------------------------------------------------------------
fn main() {
    // let args: Vec<String> = env::args().collect();
    // // 実行ファイル名を除いて、2つの引数が必要
    // if args.len() < 2 {
    //     eprintln!("Error: Missing arguments.\nUsage: {} <arg>", args[0]);
    //     std::process::exit(1); // エラー終了コードを返す
    // }
    simple_mean_online::run();
}

