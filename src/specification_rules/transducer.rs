use crate::{expr, Bother, Term, Tree};
use crate::transducer::{State, Transducer};

use crate::rendering::Rendered;

struct TransducerConfigTip {
    call_context: State,
    env: expr::Env,
    tree: Tree,
    curr: State
}

struct TransducerConfig(Vec<TransducerConfigTip>);

impl TransducerConfig {
    fn peek(&self) -> Option<&TransducerConfigTip> {
        self.0.iter().rev().peekable().peek().map(|x|*x)
    }

    // This implements the relation
    //
    // (q, E, T, r)::tl ⇒ (q', E', T', r')::tl'
    //
    // where we are given a sequence of terminals to match against.
    pub fn matches<'a>(&self, td: &Transducer, t: &'a [Term]) -> Box<dyn Iterator<Item=(TransducerConfig, &'a [Term])> +'a> {
        let tip = self.peek();
        let tip = if let Some(tip) = tip { tip } else { return None.b_iter(); };
        let r = &td.states[&tip.curr];
        unimplemented!()
    }
}
