// This is actually defined at `crate::earley::tests_for_earley`

#![allow(unused_imports)]

use crate::earley::*;
use crate::specification_rules::earley::EarleyConfig;
use crate::tests::*;
use crate::transducer::tests_for_transducer::{fig_2_a, fig_2_b, fig_2_c};
use crate::Term;

#[test]
fn imperative_fixed_width_integer() {
    let t = fig_2_a();
    let mut config = EarleyConfig::new_with_binding(Earley { transducer: t }, "n".into(), 1.into());
    dbg!(config.dyn_state());
    config.step(Term::C('1'));
    dbg!(config.dyn_state());
}
