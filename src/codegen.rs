// This is a trick: we have the source code for our tests under a single
// `src/tests/` subdirectory, but we declare it as a module *here*, under this
// module. That way, it has access to private constructors and state that a
// sibling (or in this case, nibling) module would not have access to.
#[cfg(test)]
#[path = "tests/codegen.rs"]
pub(crate) mod tests_for_codegen;
