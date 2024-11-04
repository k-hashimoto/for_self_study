use plotters::prelude::*;

struct mcmc_results {
    results: Vec<mcmc_result>,
    chains: Vec<usize>
}

struct mcmc_result {
    iter_num: Vec<usize>,
    generated_samples: Vec<f64>
}

trait VisualizeSignleResult {
    fn new(&self);
    fn store(Vec<usize>: &iter_num, Vec<f64>: &generated_samples);
}
impl Visualize for VisualizeSignleResult {
    fn new(&self) {
        self.mcmc_results.generated_samples = Vec::new();
        self.mcmc_results.iter_num = Vec::new();
    }
    fn store(Vec<usize>: &iter_num, Vec<f64>: &generated_samples) {
        self.mcmc_results.generated_samples.push(&generated_samples);
        self.mcmc_results.iter_num.push(&iter_num);
    }
}

pub fn visualizer() {
    mcmc_results{}
}

#[allow(dead_code)]
impl Visualize for mcmc_results {
    fn new(&self) {
        self.results.iter().new();
    }
    fn store(usize: &iter_num, f64: &generated_sample){
        self.store(&iter_num, &generated_sample);
    }

    // fn traceplot(){
    //     // 画像のサイズと出力ファイルを設定
    //     let root = BitMapBackend::new("./plots/bays_packages/traceplot.png", (640, 480)).into_drawing_area();
    //     root.fill(&WHITE)?;

    //     // グラフエリアの設定
    //     let mut chart = ChartBuilder::on(&root)
    //         .caption("MCMC traceplit", ("sans-serif", 50).into_font())
    //         .margin(20)
    //         .x_label_area_size(30)
    //         .y_label_area_size(30)
    //         .build_cartesian_2d(0..6, 0..60)?;

    //     chart.configure_mesh().draw()?;

    //     // 1つ目のデータ系列をプロット
    //     chart.draw_series(LineSeries::new(
    //         x.iter().zip(y1.iter()).map(|(&x, &y)| (self., y)),
    //         &BLUE,
    //     ))?
    //     .label("Series 1")
    //     .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    //     // 2つ目のデータ系列をプロット
    //     chart.draw_series(LineSeries::new(
    //         x.iter().zip(y2.iter()).map(|(&x, &y)| (x, y)),
    //         &RED,
    //     ))?
    //     .label("Series 2")
    //     .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // }
}