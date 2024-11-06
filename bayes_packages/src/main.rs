mod modules;
mod simple_linear_regiression;
mod simple_linear_regiression_filein;
mod simple_mean_bulk;
mod simple_mean_online;
mod simple_poisson;

use std::collections::HashMap;
use std::env;

// ---------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    // 実行ファイル名を除いて、2つの引数が必要
    if args.len() < 2 {
        eprintln!("Error: Missing arguments.\nUsage: {} <arg>", args[0]);
        std::process::exit(1); // エラー終了コードを返す
    }
    let command = &args[1];

    // 計算プランを登録
    let mut calculation_plans: HashMap<&str, fn() -> Result<(), Box<dyn std::error::Error>>> =
        HashMap::new();

    calculation_plans.insert("--simple_mean_online", simple_mean_online::run);
    calculation_plans.insert("--simple_mean_bulk", simple_mean_bulk::run);
    calculation_plans.insert("--simple_poisson", simple_poisson::run);
    calculation_plans.insert(
        "--simple_linear_regiression",
        simple_linear_regiression::run,
    );
    calculation_plans.insert(
        "--simple_linear_regiression_filein",
        simple_linear_regiression_filein::run,
    );

    // 引数として渡ってきたものを実行する
    match calculation_plans.get(command.as_str()) {
        Some(task) => {
            println!("Executing command: {}", command);
            task()?
        }
        None => eprintln!("Error: commands '{}' not found.", command),
    }
    Ok(())
}
