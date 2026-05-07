#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Must never panic, only ever return Err for malformed input.
    let _ = airml_tune::histogram_from_bytes(data);
});
