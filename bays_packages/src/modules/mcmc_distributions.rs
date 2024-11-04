use statrs::distribution::{Continuous, Discrete, Normal as StatrsNormal, Poisson as StatrsPoisson};

// ---------------------------------------------------------------------
// 事前、事後分布
// いずれは正規分布だけではなく、別の分布も実装するかもなので分布は以下の形式で管理して、MCMCのコードからその実装を隠蔽する
#[allow(dead_code)]
pub trait BaysDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn print(&self);
    fn update_parameters(&mut self, new_parameters: &Vec<f64>);
}

pub struct BaysNormalDistribution {
    mu: f64,
    sigma: f64,
}

#[allow(dead_code)]
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

pub struct BaysPoissonDistribution {
    lambda: f64,
}

#[allow(dead_code)]
impl BaysDistribution for BaysPoissonDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let poisson = StatrsPoisson::new(self.lambda).unwrap();
        poisson.pmf(x as u64)
    }

    fn print(&self) {
        println!("parameter = {}", self.lambda);
    }

    fn update_parameters(&mut self, new_parameters: &Vec<f64>) {
        self.lambda = new_parameters[0];
    }
}


#[allow(dead_code)]
pub fn normal(mu: f64, sigma: f64) -> Box<dyn BaysDistribution>{
    Box::new(
        BaysNormalDistribution{ mu: mu, sigma: sigma }
    )
}

pub fn poisson(lambda: f64) -> Box<dyn BaysDistribution>{
    Box::new(
        BaysPoissonDistribution{ lambda: lambda }
    )
}

// ---------------------------------------------------------------------
// 分布を取り出すインターフェイス関数
pub fn get_distribution(name: &str, parameters: &Vec<f64>) -> Box<dyn BaysDistribution> {
    if name == "normal" {
        normal(parameters[0], parameters[1])
    }else if name == "poisson" {
        poisson(parameters[0])
    } else {
        //normal(parameters[0], parameters[1]) //暫定
        panic!("Unknown distribution!")
    }
}
// fn update_parameter_vector(parameters: &mut Vec<f64>, new_values: &[f64]) {
//     let len = parameters.len().min(new_values.len()); // 更新できる範囲の長さを決定

//     for i in 0..len {
//         parameters[i] = new_values[i];
//     }
// }