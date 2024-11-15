// use core::net;
// use std::{collections::btree_map::Values, ffi::VaList};
use rand::{distributions::weighted, Rng};
type Values = Vec<f64>;

trait Layer {
    fn forward(&self, input: Values) -> Values;
//    fn backward(&mut self, grandient: Values);
}

// 全結合層
struct DenseLayer {
    weights : Vec<Values>,
    baises: Values,
}
impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        // ランダム初期化
        fn init_baises(size: usize) -> Values {
            let mut rng = rand::thread_rng();
            (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }

        fn init_weights(input_size: usize, output_size: usize) -> Vec<Values> {
            let mut rng = rand::thread_rng();
            (0..input_size)
                .map(|_| (0..output_size).map(|_| rng.gen_range(-0.1..0.1)).collect() )
                .collect()
        }

        Self {
            weights: init_weights(input_size, output_size), // output_size x input_sizeの行列
            baises:init_baises(output_size),
        }
    }

    fn sandbox(&self) {
        for x in self.baises.iter() {
            println!("{}", x);
        }
    }

}

impl Layer for DenseLayer {
    fn forward(&self, input: Values) -> Values {
        self
            .weights
            .iter()
            .map(
                |row| {
                    row.iter()
                    .zip(&input)
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                }
            )
            .zip(&self.baises)
            .map(|(weighted_sum, bias)| weighted_sum + bias)
            .collect()
    }

    // fn backward(&mut self, grandient: Values) -> Values {
    //     // dummy
    // }
}

struct Network {
    layers: Vec<Box<dyn Layer>>, // 層のリスト
}

impl Network {
    fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    fn forward(&self, input: Values) -> Values {
        self.layers
            .iter()
            .fold(input, |input, layer| layer.forward(input))
    }
}

fn main()
{
    let layer1 = Box::new(DenseLayer::new(3,4));
    let layer2 = Box::new(DenseLayer::new(4,2));

    let network = Network::new(vec![layer1, layer2]);

    let input_data = vec![1.0, 0.5, -1.5];

    let output = network.forward(input_data);

    println!("ネットワーク出力:{:?}", output);

}