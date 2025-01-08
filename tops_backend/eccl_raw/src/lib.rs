#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(nonstandard_style)]
#![allow(unused_imports)]
use libloading;
pub mod sys;
pub use sys::*;

/// Wrapper around [sys::ecclResult_t].
#[derive(Clone, PartialEq, Eq)]
pub struct EcclError(pub sys::ecclResult_t);

impl std::fmt::Debug for EcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EcclError")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum EcclStatus {
    Success,
    InProgress,
    NumResults,
}

impl sys::ecclResult_t {
    /// Transforms into a [Result] of [EcclError]
    pub fn result(self) -> Result<EcclStatus, EcclError> {
        match self {
            sys::ecclResult_t::ecclSuccess => Ok(EcclStatus::Success),
            sys::ecclResult_t::ecclInProgress => Ok(EcclStatus::InProgress),
            sys::ecclResult_t::ecclNumResults => Ok(EcclStatus::NumResults),
            _ => Err(EcclError(self)),
        }
    }
}

pub unsafe fn lib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let lib_name = "libeccl.so";
        if let Ok(lib) = Lib::new(lib_name) {
            return lib;
        } else {
            panic!("eccl library {} not found", lib_name);
        }
    })
}
