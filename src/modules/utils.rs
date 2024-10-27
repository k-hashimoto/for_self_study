#[allow(dead_code)]
pub fn mean_normal_dist(vector: &Vec<f64>) -> f64 {
    vector.iter().sum::<f64>() / vector.len() as f64
}

#[allow(dead_code)]
pub fn stddev_normal_dist(vector: &Vec<f64>) -> f64 {
    let mean = mean_normal_dist(vector);
    vector
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        .sqrt()
        / vector.len() as f64
}

#[allow(dead_code)]
pub fn hadamard_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    // アダマール積
    vec1.iter().zip(vec2.iter()).map(|(&a, &b)| a * b).collect()
}

#[allow(dead_code)]
pub fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(&a, &b)| a * b).sum()
}

#[allow(dead_code)]
pub fn subtract_constant(vec: &mut Vec<f64>, constant: f64) {
    for value in vec.iter_mut() {
        *value -= constant;
    }
}
