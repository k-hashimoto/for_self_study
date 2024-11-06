use plotters::prelude::*;
use rand::Rng;
use rand_distr::Normal;
use statrs::distribution::{Continuous, Normal as StatrsNormal};

struct BayesianLinearRegression {
    alpha_prior: StatrsNormal,
    beta_prior: StatrsNormal,
    sigma_prior: StatrsNormal,
}
impl BayesianLinearRegression {
    fn new(alpha_prior: StatrsNormal, beta_prior: StatrsNormal, sigma_prior: StatrsNormal) -> Self {
        BayesianLinearRegression {
            alpha_prior,
            beta_prior,
            sigma_prior,
        }
    }

    fn likelihood(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64, sigma: f64) -> f64 {
        // 尤度の計算。推定値とデータ点の差分が実現する確率を計算
        let mut likelihood = 1.0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let mean = alpha + beta * xi;
            let dist = StatrsNormal::new(mean, sigma).unwrap();
            likelihood *= dist.pdf(yi);
        }
        likelihood
    }
    fn posterior(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64, sigma: f64) -> f64 {
        self.likelihood(x, y, alpha, beta, sigma)
            * self.alpha_prior.pdf(alpha)
            * self.beta_prior.pdf(beta)
            * self.sigma_prior.pdf(sigma)
    }

    fn metropolis_hastings(&self, x: &[f64], y: &[f64], iterations: usize) -> Vec<(f64, f64, f64)> {
        let mut rng = rand::thread_rng();
        // 線形回帰モデルの各種パラメータの初期値
        let mut alpha: f64 = rng.gen_range(-1.0..1.0);
        let mut beta: f64 = rng.gen_range(-1.0..1.0);
        let mut sigma: f64 = rng.gen_range(0.1..2.0);

        let mut samples = vec![];
        for _ in 0..iterations {
            // 各種パラメータをランダムに"少しだけ"動かす
            let new_alpha = alpha + rng.sample(Normal::new(0.0, 1.0).unwrap());
            let new_beta = beta + rng.sample(Normal::new(0.0, 1.0).unwrap());
            let new_sigma = (sigma + rng.sample(Normal::new(0.0, 0.1).unwrap())).abs();

            // 尤度の計算とベイズ更新
            let current_posterior = self.posterior(x, y, alpha, beta, sigma);
            let proposed_posterior = self.posterior(x, y, new_alpha, new_beta, new_sigma);

            // 受け入れ判定
            if proposed_posterior / current_posterior > rng.gen::<f64>() {
                alpha = new_alpha;
                beta = new_beta;
                sigma = new_sigma;
            }

            samples.push((alpha, beta, sigma));
        }
        samples
    }
}

fn plot_results(
    x: &[f64],
    y: &[f64],
    alpha: f64,
    beta: f64,
    samples: &[(f64, f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(
        "./plots/bayes_packages/bayesian_regression_plot.png",
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

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // 要素数を変更
    let y = vec![2.1, 2.9, 3.7, 4.2, 5.1, 5.8, 6.1, 8.1, 11.4, 12.1]; // 要素数を変更

    let iterations = 10000;

    let model = BayesianLinearRegression::new(
        StatrsNormal::new(0.0, 1.0).unwrap(),
        StatrsNormal::new(0.0, 1.0).unwrap(),
        StatrsNormal::new(0.0, 1.0).unwrap(),
    );

    let samples = model.metropolis_hastings(&x, &y, iterations);
    let (alpha, beta, _sigma) = samples.last().unwrap();
    // プロットを作成
    plot_results(&x, &y, *alpha, *beta, &samples)?;

    println!("Plot saved as bayesian_regression_plot.png");
    Ok(())
}
