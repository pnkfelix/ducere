use super::*;

mod grammar;
mod luthor;

// You might think there should be a `mod tranducer;` here, but you really want
// to look at `crate::transducer::tests_for_transducer` for that.

pub(crate) fn input(s: &str) -> Vec<Term> {
    s.chars().map(|c| Term::C(c)).collect()
}
