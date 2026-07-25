//! Deterministic, dependency-free structure-field extraction primitives.

mod field;
mod fusion;

pub(crate) use field::extract_structure_field;
pub(crate) use fusion::edge_local_structure_fusion_x4;
