#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{
    debug::{self, EXIT_SUCCESS},
    hprintln,
};
use panic_semihosting as _;

use generated::predict;

// Zero input -- replace with real test data for meaningful results
static INPUT: [[[f32; 28]; 28]; 1] = [[[0.0; 28]; 28]; 1];

#[entry]
fn main() -> ! {
    hprintln!("Running inference on Cortex-M4...");

    let output = predict(&INPUT);
    let predicted = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    hprintln!("Predicted class: {}", predicted);

    hprintln!("Logits:");
    for (i, val) in output.iter().enumerate() {
        hprintln!("  [{}] {:.4}", i, val);
    }

    debug::exit(EXIT_SUCCESS);
    loop {}
}
