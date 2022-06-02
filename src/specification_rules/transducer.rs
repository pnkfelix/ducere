use crate::{expr, NonTerm, Term, Tree};
use crate::transducer::{State, StateData, Transducer, Action};

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct TransducerConfigFrame {
    call_context: State,
    env: expr::Env,
    tree: Tree,
    curr: State
}

impl TransducerConfigFrame {
    fn with_curr(&self, new_curr: State) -> Self {
        let mut n = self.clone();
        n.curr = new_curr;
        n
    }

    fn with_term(&self, term: Term) -> Self {
        let mut n = self.clone();
        n.tree.extend_term(term);
        n
    }

    fn with_bind(&self, var: expr::Var, val: expr::Val) -> Self {
        let mut n = self.clone();
        n.tree.extend_bind(var.clone(), val.clone());
        n.env.extend(var, val);
        n
    }
}


#[cfg(test)]
impl TransducerConfigFrame {
    pub(crate) fn tree(&self) -> &Tree {
        &self.tree
    }
}

impl TransducerConfigFrame {
    fn match_term(&mut self, term: Term, new_state: State) {
        self.tree.extend_term(term);
        self.curr = new_state;
    }

    fn match_pred(&mut self, expr: &expr::Expr, new_state: State) {
        let _ = expr; // we don't use this but its expected to be useful for debugging later.
        self.curr = new_state;
    }

    fn match_bind(&mut self, x: expr::Var, expr: &expr::Expr, val: expr::Val, new_state: State) {
        let _ = expr; // we don't use this but its expected to be useful for debugging later.
        self.env.extend(x.clone(), val.clone());
        self.tree.extend_bind(x, val);
        self.curr = new_state;
    }

    fn call(state: State, y_0: expr::Var, expr: &expr::Expr, val: expr::Val) -> TransducerConfigFrame {
        let _ = expr; // we don't use this but its expected to be useful for debugging later.
        TransducerConfigFrame {
            call_context: state,
            env: expr::Env::bind(y_0, val),
            tree: Tree(vec![]),
            curr: state,
        }
    }

    fn match_return(&mut self, x: Option<expr::Var>, t: Tree, a: NonTerm, v: expr::Val, new_state: State) {
        if let Some(x) = x.clone() {
            self.env.extend(x, (&t.leaves()[..]).into());
        }
        self.tree.extend_parsed(x, a, v, t);
        self.curr = new_state;
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TransducerConfig(pub(crate) Vec<TransducerConfigFrame>);

impl TransducerConfig {
    // FIXME: the (default) start state should be a property attached the
    // transducer itself
    pub fn fresh(start: State, env: expr::Env) -> Self {
        TransducerConfig(vec![TransducerConfigFrame {
            call_context: start,
            env: env,
            tree: Tree(vec![]),
            curr: start,
        }])
    }

    fn unspool(mut self) -> Option<(TransducerConfigFrame, Self)> {
        let tip = self.0.pop();
        tip.map(|t| (t, self))
    }

    fn respool(mut self, c: TransducerConfigFrame) -> Self {
        self.0.push(c);
        self
    }

    pub fn goto(self, s: State) -> Self {
        self.map_tip(|frame|frame.with_curr(s))
    }

    pub fn term(self, t: Term) -> Self {
        self.map_tip(|frame|frame.with_term(t))
    }

    pub fn bind(self, x: expr::Var, v: expr::Val) -> Self {
        self.map_tip(|frame|frame.with_bind(x, v))
    }

    pub fn call(self, s: State, input: Option<(expr::Var, &expr::Expr, expr::Val)>) -> Self {
        if let Some((y_0, expr, val)) = input {
            self.respool(TransducerConfigFrame::call(s, y_0, expr, val))
        } else {
            self.respool(TransducerConfigFrame::call(s, expr::y_0(), &().into(), ().into()))
        }
    }

    pub fn return_to(self, x: Option<expr::Var>, tree: Tree, nt: NonTerm, v: expr::Val, next: State) -> Self {
        let mut s = self;
        s.0.pop();
        s = s.map_tip(|mut frame| {
            // dbg!((&frame.tree, &v));
            frame.tree.extend_parsed(x.clone(), nt, v, tree.clone()); 
            // dbg!(&frame.tree);
            if let Some(x) = x {
                let leaves: expr::Val = tree.leaves().as_slice().into();
                dbg!(&leaves);
                frame.env.extend(x, leaves);
            }
            frame.with_curr(next)
        });
        s
    }


}

impl TransducerConfig {
    pub(crate) fn map_tip(self, f: impl FnOnce(TransducerConfigFrame) -> TransducerConfigFrame) -> Self {
        if let Some((tip, s)) = self.unspool() {
            s.respool(f(tip))
        } else {
            TransducerConfig(vec![])
        }
    }
}

#[cfg(test)]
impl TransducerConfig {
    pub(crate) fn peek(&self) -> &TransducerConfigFrame {
        self.0.iter().rev().peekable().peek().map(|x|*x).unwrap()
    }

    pub fn state(&self) -> State {
        self.peek().curr
    }
}

impl Transducer {
    // This implements the relation
    //
    // (q, E, T, r)::tl ⇒ (q', E', T', r')::tl'
    //
    // where we are given a sequence of terminals to match against.
    pub fn matches(&self, c: TransducerConfig, ts: &[Term]) -> Vec<(TransducerConfig, usize)> {
        let mut accum = Vec::new();
        let (tip, tail) = if let Some(tip_tail) = c.clone().unspool() { tip_tail } else {
            return accum;
        };
        let r: &StateData = self.data(tip.curr);

        // Basic transitions: S-Term, S-Pred, S-Bind, S-\phi
        for &(ref action, state) in r.transitions() {
            match action {
                &Action::Term(ref t) => {
                    if ts.get(0) == Some(t) {
                        let mut tip = tip.clone();
                        tip.match_term(t.clone(), state);
                        accum.push((tail.clone().respool(tip), 1));
                    } else {
                        // term doesn't match; transition cannot fire here.
                    }
                }
                Action::Constraint(e) => {
                    if e.eval(&tip.env) == expr::Val::Bool(true) {
                        let mut tip = tip.clone();
                        tip.match_pred(&e, state);
                        accum.push((tail.clone().respool(tip), 0));
                    } else {
                        // expr untrue in this env; transition cannot fire here.
                    }
                }
                &Action::Binding(ref x, ref e) => {
                    assert_ne!(x, &expr::y_0());
                    let val = e.eval(&tip.env);
                    let mut tip = tip.clone();
                    tip.match_bind(x.clone(), e, val, state);
                    accum.push((tail.clone().respool(tip), 0));
                }
                Action::Blackbox(_bb, e) => {
                    let _val = e.eval(&tip.env);
                    unimplemented!();
                }
                Action::NonTerm(..) => {
                    // this is not relevant for the inspection of tip.curr. It
                    // *will* matter when we consider if `r` is a final state
                    // and thus should cause the S-Return rule to fire.
                }
            }
        }


        // Call transitions, i.e. S-Call
        for &(ref expr, state) in r.calls() {
            let val = expr.eval(&tip.env);
            let callee = self.data(state);
            let y_0 = callee.formal_param().cloned().unwrap_or(expr::y_0());
            let next = TransducerConfigFrame::call(state, y_0, &expr, val);
            accum.push((c.clone().respool(next), 0));
        }

        // Return transitions, i.e. S-Return
        // these only match
        // if the current state is final, and if there is a frame to return to,
        if let (Some(returning), Some((caller, tail2))) = (r.output_if_final(), tail.clone().unspool())
        {
            let rr: &StateData = self.data(caller.curr);
            for &(ref action, state) in rr.transitions() {
                let (x, nt, e) = match action {
                    Action::NonTerm(x, nt, e) if returning.contains(nt) => (x, nt, e),
                    _ => continue,
                };
                let v = if let Some(e) = e { e.eval(&caller.env) } else { ().into() };
                let y_0 = self.data(tip.call_context).formal_param().cloned().unwrap_or(expr::y_0());
                if Some(&v) != tip.env.lookup(&y_0) {
                    continue;
                }
                assert_ne!(x, &Some(y_0));

                let mut caller = caller.clone();
                caller.match_return(x.clone(), tip.tree.clone(), nt.clone(), v, state);
                accum.push((tail2.clone().respool(caller), 0));
            }
        };

        return accum;
    }
}
