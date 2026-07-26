//! Preflight check for Ascend NPU runtime version.
//!
//! Calls `aclrtGetVersion` via the pegaflow-core device abstraction to
//! verify the CANN version is at least 8.5.0 (minimum required for
//! IPC and async memory copy APIs).

use log::info;
use pegaflow_core::device::ascend;

/// Error returned when the Ascend version preflight check fails.
pub(crate) struct AscendVersionError(String);

impl std::fmt::Display for AscendVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Debug for AscendVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for AscendVersionError {}

/// Minimum CANN runtime version required.  `aclrtGetVersion` returns the
/// *runtime* version of the CANN stack, not the toolkit packaging version.
/// For CANN toolkit 8.5.0, the runtime version is 1.16.0.
/// For CANN toolkit 8.5.1, the runtime version may be 1.16.0 or 1.16.x.
const MIN_MAJOR: i32 = 1;
const MIN_MINOR: i32 = 16;
const MIN_PATCH: i32 = 0;

/// Run the Ascend runtime version preflight check.
///
/// Returns `Ok(())` if CANN >= 8.5.0 and the runtime library can be loaded.
/// Returns `Err(AscendVersionError)` on version mismatch or loading failure.
pub(crate) fn preflight() -> Result<(), AscendVersionError> {
    info!(
        "Ascend preflight: minimum required version {}.{}.{}",
        MIN_MAJOR, MIN_MINOR, MIN_PATCH
    );

    let (major, minor, patch) = ascend::get_acl_version().map_err(|err| {
        AscendVersionError(format!(
            "Ascend preflight failed: unable to query CANN runtime version: {err}. \
             Ensure libascendcl.so is in LD_LIBRARY_PATH and the Ascend driver is loaded."
        ))
    })?;

    info!(
        "Ascend preflight: runtime version {}.{}.{} detected",
        major, minor, patch
    );

    if !is_compatible_ascend_version(major, minor, major, minor, patch) {
        return Err(AscendVersionError(format!(
            "Ascend version mismatch: minimum required {}.{}.{}, detected {}.{}.{}. \
             Upgrade CANN toolkit to >= 8.5.0.",
            MIN_MAJOR, MIN_MINOR, MIN_PATCH, major, minor, patch
        )));
    }

    Ok(())
}

/// Check whether the detected CANN version meets the minimum requirement.
///
/// The `build_major`/`build_minor` parameters are reserved for future use
/// when compile-time version detection is added. Currently they are always
/// equal to the runtime version.
fn is_compatible_ascend_version(
    _build_major: i32,
    _build_minor: i32,
    runtime_major: i32,
    runtime_minor: i32,
    _runtime_patch: i32,
) -> bool {
    // Version comparison: must be >= MIN_MAJOR.MIN_MINOR
    if runtime_major > MIN_MAJOR {
        return true;
    }
    if runtime_major == MIN_MAJOR && runtime_minor >= MIN_MINOR {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_version_equal_to_minimum() {
        assert!(is_compatible_ascend_version(8, 5, 8, 5, 0));
    }

    #[test]
    fn compatible_version_higher_minor() {
        assert!(is_compatible_ascend_version(8, 5, 8, 7, 0));
    }

    #[test]
    fn compatible_version_higher_major() {
        assert!(is_compatible_ascend_version(8, 5, 9, 0, 0));
    }

    #[test]
    fn incompatible_version_lower_minor() {
        assert!(!is_compatible_ascend_version(8, 5, 8, 4, 0));
    }

    #[test]
    fn incompatible_version_lower_major() {
        assert!(!is_compatible_ascend_version(8, 5, 7, 0, 0));
    }
}
