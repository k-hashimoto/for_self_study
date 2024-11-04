use plotters::prelude::*;
use prettytable::{Table, Row, Cell};

pub fn trace_plot(chains: &Vec<Vec<f64>>, x_min: usize, x_max: usize, y_min: f64, y_max: f64) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("./plots/bays_packages/trace_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Trace Plot of MCMC Samples for 4 Chains(blue:1, red:2, green:3, purple:4)", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // チェーンごとに異なる色でトレースプロットを描画
    // let colors = [full_palette::BLUE, full_palette::RED, full_palette::GREEN, full_palette::PURPLE];
    // チェーンごとの色を透明度付きで設定
    let colors = [
        RGBAColor(0, 0, 255, 0.2),    // 半透明の青
        RGBAColor(255, 0, 0, 0.2),    // 半透明の赤
        RGBAColor(0, 255, 0, 0.2),    // 半透明の緑
        RGBAColor(128, 0, 128, 0.2),  // 半透明の紫
    ];
    for (i, chain) in chains.iter().enumerate() {
        chart.draw_series(
                LineSeries::new(
                    chain.iter().enumerate().map(|(j, &value)| (j, value)),
                    &colors[i],
                )
        )?;
    }

    // 保存処理の完了
    root.present()?;
    println!("Trace plot saved as 'plots/bays_packages/trace_plot.png'");
    Ok(())
}

pub fn print_mcmc_summary_table(true_mean: &f64, posterior_mean: &f64, lower_bound: &f64, upper_bound: &f64){
    let mut table = Table::new();
    table.add_row(Row::new(vec![
        Cell::new("項目"),
        Cell::new("説明"),
    ]));
    table.add_row(Row::new(vec![
        Cell::new("観測値の平均"),
        Cell::new(&format!("{:.3}", true_mean)),
    ]));

    table.add_row(Row::new(vec![
        Cell::new("推定結果"),
        Cell::new(&format!("{:.3}", posterior_mean)),
    ]));

    table.add_row(Row::new(vec![
        Cell::new("事後分布の平均値 - 単純平均"),
        Cell::new(&format!("{:.3}", posterior_mean - true_mean)),
    ]));
    table.add_row(Row::new(vec![
        Cell::new("95% Credible Interval"),
        Cell::new(&format!("{:.3} ~ {:.3}", lower_bound, upper_bound)),
    ]));
    // テーブルを表示
    table.printstd();
}