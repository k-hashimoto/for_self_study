type Values = Vec<f64>;

trait Layer {
    fn forward(&self, input: Values) -> Values;
    fn backward(&mut self, grandient: Values);
}

// 全結合層
struct DenseLayer {
    weights : Vec<Values>,
    baises: Values,
}
impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: vec![vec![0.0; output_size]; input_size], // output_size x input_sizeの行列
            baises: vec![0.0; output_size],
        }
    }
}

// impl Layer for DenseLayer {
//     fn forward(&self, input: Values) -> Values {
//         // dummy
//     }

//     fn backward(&mut self, grandient: Values) -> Values {
//         // dummy
//     }
// }

fn main()
{
    let dense = DenseLayer::new(2,2);
    println!("{:?}", dense.weights);
}