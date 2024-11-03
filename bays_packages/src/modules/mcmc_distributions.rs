use statrs::distribution::{Continuous, Normal as StatrsNormal};

// ---------------------------------------------------------------------
// 事前、事後分布
// いずれは正規分布だけではなく、別の分布も実装するかもなので分布は以下の形式で管理して、MCMCのコードからその実装を隠蔽する
pub trait BaysDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn print(&self);
    fn update_parameters(&mut self, new_parameters: &Vec<f64>);
}
pub struct BaysNormalDistribution {
    mu: f64,
    sigma: f64,
}
impl BaysDistribution for BaysNormalDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let normal = StatrsNormal::new(self.mu, self.sigma).unwrap();
        normal.pdf(x)
    }

    fn print(&self) {
        println!("parameter = {}, {}", self.mu, self.sigma);
    }

    fn update_parameters(&mut self, new_parameters: &Vec<f64>) {
        self.mu = new_parameters[0];
        self.sigma = new_parameters[1];
    }
}

pub fn normal(mu: f64, sigma: f64) -> Box<dyn BaysDistribution>{
    Box::new(
        BaysNormalDistribution{ mu: mu, sigma: sigma }
    )
}

// ---------------------------------------------------------------------
// 分布を取り出すインターフェイス関数
pub fn get_distribution(name: &str, parameters: &Vec<f64>) -> Box<dyn BaysDistribution> {
    if name == "normal" {
        normal(parameters[0], parameters[1])
    } else {
        normal(parameters[0], parameters[1]) // 暫定的
    }
}
// fn update_parameter_vector(parameters: &mut Vec<f64>, new_values: &[f64]) {
//     let len = parameters.len().min(new_values.len()); // 更新できる範囲の長さを決定

//     for i in 0..len {
//         parameters[i] = new_values[i];
//     }
// }