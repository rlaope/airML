//! Command implementations
//!
//! Contains the implementation for each CLI subcommand.

pub mod bench;
pub mod generate;
pub mod info;
pub mod install_runtime;
pub mod pull;
pub mod run;
pub mod serve;
pub mod system;

#[cfg(feature = "nlp")]
pub mod embed;

pub use bench::execute as bench;
pub use generate::execute as generate;
pub use info::execute as info;
pub use install_runtime::execute as install_runtime;
pub use pull::execute as pull;
pub use run::execute as run;
pub use serve::execute as serve;
pub use system::execute as system;

#[cfg(feature = "nlp")]
pub use embed::execute as embed;
