#![no_std]
#![no_main]

use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m_rt::entry;
use panic_halt as _;

use generated::predict;

// Zero input -- replace with real data in your firmware.
static INPUT: [[[f32; 4]; 1]; 1] = [[[0.0; 4]; 1]; 1];

// Sink to prevent the optimizer from removing the inference call.
static SINK: AtomicU32 = AtomicU32::new(0);

#[entry]
fn main() -> ! {
    let output = predict(&INPUT);
    for val in output.iter() {
        SINK.store(val.to_bits(), Ordering::SeqCst);
    }
    loop {}
}
