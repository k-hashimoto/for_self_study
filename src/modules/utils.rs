#[allow(dead_code)]
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

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

#[allow(dead_code)]
pub fn read_csv_to_two_vectors(path: &str) -> Result<(Vec<String>, Vec<f32>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut column1 = Vec::new();
    let mut column2 = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() >= 2 {
            column1.push(record[0].to_string());
            column2.push(record[1].parse::<f32>().unwrap_or(0.0));
        }
    }

    Ok((column1, column2))
}
