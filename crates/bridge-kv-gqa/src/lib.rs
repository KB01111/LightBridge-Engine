//! Bounded, page-addressed F32 grouped-query KV state.

mod cache;
mod error;

pub use cache::{PagedKvCache, KV_SNAPSHOT_FORMAT, KV_SNAPSHOT_VERSION};
pub use error::KvError;
