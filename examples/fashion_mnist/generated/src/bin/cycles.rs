#![no_std]
#![no_main]

use cortex_m::peripheral::DWT;
use cortex_m_rt::entry;
use cortex_m_semihosting::{
    debug::{self, EXIT_SUCCESS},
    hprintln,
};
use panic_semihosting as _;

use generated::predict;

static INPUT: [[[f32; 28]; 28]; 1] = [[[0.0; 28]; 28]; 1];

#[entry]
fn main() -> ! {
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    // Warmup: discard first run so cycles below reflect the
    // steady-state cost, not first-touch cache misses.
    let _ = predict(&INPUT);

    let start = DWT::cycle_count();
    let output = predict(&INPUT);
    let end = DWT::cycle_count();
    let cycles = end.wrapping_sub(start);

    hprintln!("predict cycles (DWT, QEMU approximation): {}", cycles);

    // Anchor the output so the optimizer can't elide predict.
    let predicted = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    hprintln!("predicted class: {}", predicted);

    debug::exit(EXIT_SUCCESS);
    loop {}
}
