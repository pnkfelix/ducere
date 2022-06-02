//! Traditional earley: gradually build, from left-to-right, set of "Earley
//! items" for each position in input.
//!
//! The Earley sets memoize: rather than reparse portions of input, it reuses
//! the information from the Earley sets.
//!
//! Yakker modified Earley: Indexed sets of parse trees (i.e. Forests). A tree T
//! belongs to the set `tree(i, j, q, E, r)` when that tree is computed by
//! parsing the input from position `i+1` to position `j`. The parse of this
//! subsequence must have begun with the transducer in callee state `q` and
//! ended with the transducer in state `r`. Environment `E` is the environment
//! that was built during the course of the parse.
//!
//! (but: what does it mean when i == j, such as in ET-Init, if we're talking
//! about parsing i+1 to position j? Is that a typo in the definition of `tree`
//! above?)

use crate::transducer::{Transducer};

#[cfg(test)]
#[path = "tests/earley.rs"]
mod tests_for_earley;

pub struct Earley {
    #[allow(dead_code)] 
    transducer: Transducer,
}

impl Earley {
    #[allow(dead_code)] 
    pub(crate) fn transducer(&self) -> &Transducer { &self.transducer }
}
