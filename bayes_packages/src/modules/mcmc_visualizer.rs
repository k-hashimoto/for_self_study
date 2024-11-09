use plotters::prelude::*;
use plotters::prelude::*;
use prettytable::{Cell, Row, Table};
use std::collections::HashMap; // HashMapをインポート
use std::f64::consts::PI;

// mu1_samplesのカーネル密度推定とDATAの平均を描画しPNGに保存する関数
pub fn plot_mu1_kde_and_data_mean(
    mu1_samples: &Vec<f64>,
    data: &Vec<f64>,
    output_path: &str,
    bandwidth: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // ガウシアンカーネル関数
    fn gaussian_kernel(x: f64, mu: f64, bandwidth: f64) -> f64 {
        (-0.5 * ((x - mu) / bandwidth).powi(2)).exp() / (bandwidth * (2.0 * PI).sqrt())
    }

    // カーネル密度推定を計算する関数
    fn kernel_density_estimate(samples: &[f64], x: f64, bandwidth: f64) -> f64 {
        samples
            .iter()
            .map(|&sample| gaussian_kernel(x, sample, bandwidth))
            .sum::<f64>()
            / samples.len() as f64
    }

    // 出力ファイルをPNG形式で設定
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("mu1 Samples KDE and Data Mean", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(30.0f64..70.0f64, 0.0..0.1)?;

    chart.configure_mesh().draw()?;

    // mu1_samplesのカーネル密度推定をプロット
    let kde_points: Vec<(f64, f64)> = (300..700)
        .map(|x| x as f64 / 10.0)
        .map(|x| (x, kernel_density_estimate(mu1_samples, x, bandwidth)))
        .collect();

    chart
        .draw_series(LineSeries::new(kde_points, &BLUE))?
        .label("mu1 KDE")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // data全体の平均を計算
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;

    // dataの平均値の線をプロット (赤)
    chart
        .draw_series(LineSeries::new(
            vec![(data_mean, 0.0), (data_mean, 0.1)],
            &RED,
        ))?
        .label("Data Mean")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 5, y)], &RED));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
pub fn trace_plot(
    chains: &Vec<Vec<f64>>,
    x_min: usize,
    x_max: usize,
    y_min: f64,
    y_max: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("./plots/bayes_packages/trace_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Trace Plot of MCMC Samples for 4 Chains(blue:1, red:2, green:3, purple:4)",
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // チェーンごとに異なる色でトレースプロットを描画
    // let colors = [full_palette::BLUE, full_palette::RED, full_palette::GREEN, full_palette::PURPLE];
    // チェーンごとの色を透明度付きで設定
    let colors = [
        RGBAColor(0, 0, 255, 0.2),   // 半透明の青
        RGBAColor(255, 0, 0, 0.2),   // 半透明の赤
        RGBAColor(0, 255, 0, 0.2),   // 半透明の緑
        RGBAColor(128, 0, 128, 0.2), // 半透明の紫
    ];
    for (i, chain) in chains.iter().enumerate() {
        chart.draw_series(LineSeries::new(
            chain.iter().enumerate().map(|(j, &value)| (j, value)),
            &colors[i],
        ))?;
    }

    // 保存処理の完了
    root.present()?;
    println!("Trace plot saved as 'plots/bays_packages/trace_plot.png'");
    Ok(())
}

pub fn print_mcmc_summary_table(
    true_mean: &f64,
    posterior_mean: &f64,
    lower_bound: &f64,
    upper_bound: &f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut table = Table::new();
    table.add_row(Row::new(vec![Cell::new("項目"), Cell::new("説明")]));
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
    Ok(())
}


// ----------------------------------------------------------------------------------------------
pub fn plot_results_linear_regression(
    path: &str,
    x: &[f64],
    y: &[f64],
    alpha: f64,
    beta: f64,
    samples: &[(f64, f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(
        path,
        (800, 600),
    )
    .into_drawing_area();
    root.fill(&WHITE)?;

    // Determine x and y range dynamically based on data, adding a small margin
    let x_min = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - 0.5;
    let x_max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() + 0.5;
    let y_min = *y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - 0.5;
    let y_max = *y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() + 0.5;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Bayesian Linear Regression with 95% Credible Interval",
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // Plot the original data points
    chart.draw_series(
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| Circle::new((xi, yi), 5, BLUE.filled())),
    )?;

    // Calculate 95% credible interval
    let regression_line: Vec<_> = (0..100)
        .map(|i| {
            let xi = x_min + (x_max - x_min) * i as f64 / 100.0;
            let mut predictions: Vec<f64> = samples.iter().map(|&(a, b, _)| a + b * xi).collect();
            predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Get the 2.5th percentile and 97.5th percentile for 95% credible interval
            let lower = predictions[(0.025 * predictions.len() as f64).round() as usize];
            let upper = predictions[(0.975 * predictions.len() as f64).round() as usize];
            let mean = alpha + beta * xi;
            (xi, mean, lower, upper)
        })
        .collect();

    // Draw the regression line
    chart
        .draw_series(LineSeries::new(
            regression_line.iter().map(|&(xi, mean, _, _)| (xi, mean)),
            &RED,
        ))?
        .label("Regression Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    // Draw the 95% credible interval as a filled area
    chart
        .draw_series(AreaSeries::new(
            regression_line
                .iter()
                .map(|&(xi, _, lower, _)| (xi, lower))
                .chain(
                    regression_line
                        .iter()
                        .rev()
                        .map(|&(xi, _, _, upper)| (xi, upper)),
                ),
            0.0,
            &RED.mix(0.2),
        ))?
        .label("95% Credible Interval")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED.mix(0.2)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    Ok(())
}
