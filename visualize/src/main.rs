use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // データの定義
    let x = vec![1, 2, 3, 4, 5];
    let y1 = vec![10, 20, 30, 40, 50];
    let y2 = vec![15, 25, 35, 45, 55];
    let y3 = vec![5, 15, 25, 35, 45];

    // 画像のサイズと出力ファイルを設定
    let root = BitMapBackend::new("./plots/visualize/plot_multiple.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // グラフエリアの設定
    let mut chart = ChartBuilder::on(&root)
        .caption("Sample Multi-Line Plot", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..6, 0..60)?;

    chart.configure_mesh().draw()?;

    // 1つ目のデータ系列をプロット
    chart.draw_series(LineSeries::new(
        x.iter().zip(y1.iter()).map(|(&x, &y)| (x, y)),
        &BLUE,
    ))?
    .label("Series 1")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    // 2つ目のデータ系列をプロット
    chart.draw_series(LineSeries::new(
        x.iter().zip(y2.iter()).map(|(&x, &y)| (x, y)),
        &RED,
    ))?
    .label("Series 2")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // 3つ目のデータ系列をプロット
    chart.draw_series(LineSeries::new(
        x.iter().zip(y3.iter()).map(|(&x, &y)| (x, y)),
        &GREEN,
    ))?
    .label("Series 3")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    // 凡例の表示
    chart.configure_series_labels().border_style(&BLACK).draw()?;

    Ok(())
}