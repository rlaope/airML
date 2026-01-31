//! Command implementations
//!
//! Contains the implementation for each CLI subcommand.

pub mod bench;
pub mod info;
pub mod run;
pub mod system;

#[cfg(feature = "nlp")]
pub mod embed;

pub use bench::execute as bench;
pub use info::execute as info;
pub use run::execute as run;
pub use system::execute as system;

#[cfg(feature = "nlp")]
pub use embed::execute as embed;
