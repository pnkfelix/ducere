#[macro_use] extern crate lalrpop_util;

use std::collections::HashSet;

pub trait Recognizer {
    type Term;
    type String;
    fn accept(&self, iter: &mut dyn Iterator<Item=&Self::Term>) -> Option<Self::String>;
}

#[derive(Clone, PartialEq, Eq)]
pub struct Blackbox {
    name: String,
    from_val: fn(expr::Val) -> Box<dyn Recognizer<Term=Term, String=String>>,
}

impl std::fmt::Debug for Blackbox {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, "blackbox[{}]", self.name)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Grammar { pub rules: Vec<Rule> }

impl Grammar {
    pub fn nonterms(&self) -> HashSet<NonTerm> {
        // one might argue that this should also scan the right-hand sides of
        // the rules for non-terminals that are otherwise undefined. But I say
        // that grmmars that do that deserve to be considered ill-formed.
        self.rules.iter().map(|r|r.lhs.clone()).collect()
    }
    pub fn terms(&self) -> HashSet<Term> {
        self.rules.iter().flat_map(|r|r.rhs.terms()).collect()
    }
}

impl Grammar {
    pub fn empty() -> Self { Grammar { rules: vec![] } }

    fn rule(&self, nonterm: &NonTerm) -> Option<&Rule> {
        self.rules.iter().find(|r| &r.lhs == nonterm)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Rule {
    label: String,
    lhs: NonTerm,
    param: Option<expr::Var>,
    rhs: RegularRightSide
}

impl Rule {
    fn new(lhs: NonTerm, param: Option<expr::Var>, rhs: RegularRightSide) -> Rule {
        Rule { label: String::new(), lhs, param, rhs }
    }

    fn labelled_new(label: impl Into<String>, lhs: NonTerm, param: Option<expr::Var>, rhs: RegularRightSide) -> Rule {
        Rule { label: label.into(), lhs, param, rhs }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum RegularRightSide {
    EmptyString,
    EmptyLanguage,
    Term(Term),
    #[allow(non_snake_case)]
    NonTerm { x: Option<expr::Var>, A: NonTerm, e: Option<expr::Expr> },
    Binding { x: expr::Var, e: expr::Expr },
    Concat(Box<Self>, Box<Self>),
    Either(Box<Self>, Box<Self>),
    Kleene(Box<Self>),
    Constraint(expr::Expr),
    Blackbox(Blackbox, expr::Expr),
}

impl RegularRightSide {
    fn terms(&self) -> Box<dyn Iterator<Item=Term>> {
        match self {
            RegularRightSide::EmptyString |
            RegularRightSide::EmptyLanguage |
            RegularRightSide::NonTerm { .. } |
            RegularRightSide::Binding { .. }  |
            RegularRightSide::Constraint(_) |
            RegularRightSide::Blackbox(..) => None.b_iter(),

            RegularRightSide::Term(t) => Some(t.clone()).b_iter(),

            RegularRightSide::Concat(lhs, rhs) |
            RegularRightSide::Either(lhs, rhs) => Box::new(lhs.terms().chain(rhs.terms())),

            RegularRightSide::Kleene(inner) => inner.terms(),
        }
    }
}

#[derive(Copy, Clone)]
enum RrsContext { Concat, Either, Kleene, }

impl RegularRightSide {
    fn needs_parens(&self, context: RrsContext) -> bool {
        match (self, context) {
            (RegularRightSide::EmptyString |
             RegularRightSide::EmptyLanguage |
             RegularRightSide::Term(_) |
             RegularRightSide::NonTerm { .. } |
             RegularRightSide::Binding { .. }, _) => false,

            (RegularRightSide::Concat(..), RrsContext::Concat) => false,
            (RegularRightSide::Concat(..), RrsContext::Either | RrsContext::Kleene) => true,

            (RegularRightSide::Either(..), RrsContext::Either) => false,
            (RegularRightSide::Either(..), RrsContext::Concat | RrsContext::Kleene) => true,

            (RegularRightSide::Kleene(_), _) => false,

            (RegularRightSide::Constraint(_), _) => false,
            (RegularRightSide::Blackbox(..), _) => unimplemented!(),
        }
    }
}

impl std::fmt::Display for RegularRightSide {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RegularRightSide::EmptyString => write!(w, "''"),
            RegularRightSide::EmptyLanguage => write!(w, "empty"),
            RegularRightSide::Term(Term::C(c)) => write!(w, "'{}'", c),
            RegularRightSide::Term(Term::S(s)) => write!(w, "'{}'", s),
            RegularRightSide::NonTerm { x, A: NonTerm(name), e } => {
                match (x, e) {
                    (None, None) => write!(w, "{}", name),
                    (None, Some(e)) => write!(w, "<{}({})>", name, e),
                    (Some(expr::Var(x)), None) => write!(w, "<{}:={}>", x, name),
                    (Some(expr::Var(x)), Some(e)) => write!(w, "<{}:={}({})>", x, name, e),
                }
            }
            RegularRightSide::Binding { x: expr::Var(x), e } => write!(w, "{{ {} := {} }}", x, e),
            RegularRightSide::Concat(lhs, rhs) => {
                let ctxt = RrsContext::Concat;
                if !lhs.needs_parens(ctxt) && !rhs.needs_parens(ctxt) {
                    write!(w, "{} {}", lhs, rhs)
                } else {
                    write!(w, "({}) ({})", lhs, rhs)
                }
            }
            RegularRightSide::Either(lhs, rhs) => {
                let ctxt = RrsContext::Either;
                if !lhs.needs_parens(ctxt) && !rhs.needs_parens(ctxt) {
                    write!(w, "{} | {}", lhs, rhs)
                } else {
                    write!(w, "({}) | ({})", lhs, rhs)
                }
            }
            RegularRightSide::Kleene(r) => {
                let ctxt = RrsContext::Kleene;
                if r.needs_parens(ctxt) {
                    write!(w, "{}*", r)
                } else {
                    write!(w, "({})*", r)
                }
            }
            RegularRightSide::Constraint(e) => {
                write!(w, "[{}]", e)
            }
            RegularRightSide::Blackbox(_bb, _e) => {
                unimplemented!()
            }
        }
    }
}

trait Rendered {
    fn rendered(&self) -> String;
}

impl Rendered for expr::Env {
    fn rendered(&self) -> String {
        format!("{}", self)
    }
}

impl Rendered for Option<expr::Env> {
    fn rendered(&self) -> String {
        match self {
            Some(env) => format!("{}", env),
            None => "nil".to_string(),
        }
    }
}

impl Rendered for [Term] {
    fn rendered(&self) -> String {
        self.iter()
            .map(|t| {
                match t {
                    Term::C(c) => c.to_string(),
                    Term::S(s) => s.to_string(),
                }
            })
            .collect()
    }
}

trait Bother<T> { fn b_iter(self) -> Box<dyn Iterator<Item=T>>; }

impl<T: 'static> Bother<T> for Option<T> {
    fn b_iter(self) -> Box<dyn Iterator<Item=T>> {
        Box::new(self.into_iter())
    }
}

struct Cross<I: Iterator, ToJ, J> where I: Iterator,
{
    i_iter: I,
    to_j: ToJ,
    cursor: Option<(I::Item, J)>,
}

impl<I, ToJ, J>  Cross<I, ToJ, J> where
    I: Iterator, ToJ: for<'a> FnMut(&'a I::Item) -> J, J: Iterator
{
    fn new(mut i_iter: I, mut to_j: ToJ) -> Self {
        let curr_i = i_iter.next();
        let cursor = curr_i.map(|i| {
            let js = to_j(&i);
            (i, js)
        });
        Self { i_iter, to_j, cursor }
    }
}

impl<I, ToJ, J> Iterator for Cross<I, ToJ, J> where
    I: Iterator, I::Item: Clone, ToJ: for<'a> FnMut(&'a I::Item) -> J, J: Iterator
{
    type Item = (I::Item, J::Item);
    fn next(&mut self) -> Option<(I::Item, J::Item)> {
        let (ref mut i, ref mut j_iter) = if let Some(t) = self.cursor.as_mut() { t } else { return None; };
        'advance_j: loop {
            if let Some(j) = j_iter.next() {
                return Some((i.clone(), j));
            }
            // else: curr_j_iter exhausted. Advance to next i and try that.
            let new_i = match self.i_iter.next() {
                None => { self.cursor = None; return None; }
                Some(i) => { i }
            };
            let new_j_iter = (self.to_j)(&new_i);
            *i = new_i;
            *j_iter = new_j_iter;
            continue 'advance_j;
        }
    }
}

trait ParseMatches {
    fn has_parse(&mut self) -> bool;
    fn no_parse(&mut self) -> bool { ! self.has_parse() }
}

impl<'s> ParseMatches for Box<dyn Iterator<Item=expr::Env> + 's> {
    fn has_parse(&mut self) -> bool { self.next().is_some() }
}

impl Grammar {
    // This implements the relation
    // env ⊢ w ∈ r ⇒ Env
    //
    // Notes:
    //
    //  * the given `w` has to *exactly* match the needs of the rule.
    //
    //  * there may be multiple ways for given `w` to match a rule, and that
    //  means different resulting env. The Yakker paper doesn't say much about
    //  this; my reading of the theorems, especially of Earley soundness, that
    //  if different derivations exist, then you don't know which one you will
    //  get. (After all, if you didn't allow for that, and forced the system to
    //  yield every derivation, then that would preclude most optimizations for
    //  the parser.)
    //     * revisiting this point: it is possible cannot avoid searching through
    //       multiple potential matches. One issue is constraints [PRED]: if you
    //       can get distinct environments out, you need to find the ones that will
    //       satisfy PRED.
    pub fn matches<'s>(&'s self, w: &'s [Term], r: &'s RegularRightSide) -> Box<dyn Iterator<Item=expr::Env> + 's> {
        self.matches_recur(expr::Env::empty(), w, r, 0).0
    }

    fn split_matches<'s, 'a>(&'s self, mut splits: impl Clone + Iterator<Item=usize> + 's, env: expr::Env, w: &'s [Term], mut rs: impl Clone + Iterator<Item=&'s RegularRightSide> + 's, depth: usize) -> (Box<dyn Iterator<Item=expr::Env> + 's>, usize) {
        let (split, r) = match (splits.next(), rs.next()) {
            (Some(split), Some(r)) => (split, r),
            _ => {
                assert_eq!(w.len(), 0);
                return (Some(expr::Env::empty()).b_iter(), depth);
            }
        };

        let (w_pre, w_post) = w.split_at(split);
        let (e_pre_iter, accum) = self.matches_recur(env.clone(), w_pre, &r, depth+1);
        (Box::new(e_pre_iter.flat_map(move |left_env| {
            let left_env = left_env.clone();
            self.split_matches(splits.clone(), env.clone().concat(left_env.clone()), w_post, rs.clone(), accum+1).0
                .map(move |env| left_env.clone().concat(env))
        })), accum)
    }

    fn matches_recur<'s>(&'s self, env: expr::Env, w: &'s [Term], r: &'s RegularRightSide, depth: usize) -> (Box<dyn Iterator<Item=expr::Env> + 's>, usize) {
        let accum = depth;
        let indent: String = std::iter::repeat(' ').take(depth).collect();
        println!("{}env: {} w: {:?} r: `{}`", indent, env, w.rendered(), r);

        match r {
            // GL-EPS
            RegularRightSide::EmptyString =>
                if w.len() == 0 {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                },
            // GL-TERM
            RegularRightSide::Term(t) =>
                if t.matches(w) {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                },
            // GL-PRED
            RegularRightSide::Constraint(e) =>
                if e.eval(&env) == expr::TRUE {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                }
            // GL-BIND
            RegularRightSide::Binding { x, e } =>
                if w.len() == 0 {
                    (Some(expr::Env::bind(x.clone(), e.eval(&env))).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                },
            // GL-φ
            RegularRightSide::Blackbox(bb, e) => {
                let v = e.eval(&env);
                let phi = (bb.from_val)(v);
                // The BB interface is iterator based; but the spec for GL-φ
                // implies that the whole string must be matched. We ensure this
                // by checking that the iterator was exhausted by the blackbox.
                //
                // (This is clearly a suboptimal interface overall; i.e. this
                // system will potentially invoke the same blackbox repeatedly,
                // when what we *should* do is have a blackbox interface that
                // allows it to incrementally process more of the input on
                // demand, and signal recognition on successive prefixes.
                //
                // In any case, in general the Iterator spec does not guarantee
                // that successive calls to `next()` will return `None`; but
                // calling `fuse()` *will* ensure this.
                let mut cs = w.iter().fuse();
                let accepted = phi.accept(&mut cs);
                if accepted.is_some() && cs.next().is_none() {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                }
            }
            // GL-A
            RegularRightSide::NonTerm { x, A, e } => {
                let Rule { label: _ , lhs: _a, param: opt_var, rhs: subrule } =
                    self.rule(A).unwrap();
                let subenv = match (e, opt_var) {
                    (None, None) =>
                        // great: non-parameterized terminals don't need values
                        expr::Env::empty(),
                    (Some(e), Some(y_0)) => {
                        let v = e.eval(&env);
                        expr::Env::bind(y_0.clone(), v)
                    }
                    (Some(e), None) => {
                        // not as great: I'd prefer to not use y_0 formalism.
                        let v = e.eval(&env);
                        expr::Env::bind(expr::y_0(), v)
                    }
                    (None, Some(y_0)) => {
                        panic!("provided expr argument {:?} \
                                to *unparameterized* non-term {:?}({:?})",
                               e, _a, y_0);
                    }
                };
                let (subresults, accum) = self.matches_recur(subenv, w, subrule, depth+1);
                let x = x.clone();
                (Box::new(subresults.map(move |_discarded_env| {
                    match &x {
                        Some(x) => expr::Env::bind(x.clone(), w.into()),
                        None => expr::Env::empty(),
                    }
                })), accum)
            }
            // GL-SEQ
            //
            // this is where we *really* see the weakness of the API
            // being used here: instead of streaming through the input
            // and having some way to save intermediate results,
            // this code is forced to try each partition of the strings,
            // (including empty strings!).
            RegularRightSide::Concat(r1, r2) => {
                for i in 0..=w.len() {
                    let wr = w.rendered();
                    let (w1,  w2) = w.split_at(i);
                    println!("{}`{:?}` in Concat(`{}`,`{}`) trial i={} yields {:?} {:?}",
                             indent, wr, r1, r2, i, w1.rendered(), w2.rendered());
                    let (subresults_1, accum) = self.matches_recur(env.clone(), w1, r1, depth+1);
                    let cross = Cross::new(
                        subresults_1,
                        |env1| {
                            let new_env = env.clone().concat(env1.clone());
                            let (subresults_2, accum) = self.matches_recur(new_env, w2, r2, accum+1);
                            // println!("{}i: {} env2: {} w2: {:?} r2: `{}`, w2_in_r2: {:?}", indent, i, new_env, w2.rendered(), r2, format!("{}", &env2));
                            subresults_2.map(move |sr| (sr, accum))
                        });
                    for (env1, (env2, accum)) in cross {
                        println!("{}i: {} env: {} w1: {:?} r1: `{}` env1: {} w2: {:?} r2: `{}` env2: {}",
                                 indent, i, env,
                                 w1.rendered(), r1, env1.rendered(),
                                 w2.rendered(), r2, env2.rendered(),
                        );
                        return (Some(env1.concat(env2)).b_iter(), accum);
                    }
                }
                return (None.b_iter(), accum);
            }
            // GL-ALTL + GL-ALTR
            RegularRightSide::Either(r1, r2) => {
                let (result1, accum) = self.matches_recur(env.clone(), w, r1, depth+1);
                let (result2, accum) = self.matches_recur(env, w, r2, accum+1);
                return (Box::new(result1.chain(result2)), accum);
            }
            // GL-*
            //
            // Remember that note up above about seeing the weakness of the API
            // on GL-SEQ, because it needs to just guess the split point to use?
            // This is in theory worse, since this rule as stated is
            // parameterized over any choice of k splits of the string, where
            // *empty strings are permitted*. and where the environment is being
            // built up as we go.
            //
            // But, in practice, we need to keep in mind that the whole goal is
            // to process the input string. I haven't yet figured out whether
            // the constraint language could force one to match an arbitrary
            // number of empty strings, but I'm not going to try to handle that
            // case here for now. Instead, I'm going to assume that for Kleene
            // closure that every substring of a non-empty input *is* non-empty.
            RegularRightSide::Kleene(sr) => {
                // if the string is empty, we trivially match
                if w.len() == 0 {
                    println!("{}Kleene trivial empty match env: {} w: {:?} r: `{}`",
                             indent, env, w.rendered(), r);
                    return (Some(expr::Env::empty()).b_iter(), accum);
                }

                let r = sr;
                // otherwise, we will enumerate the ways to break up the string.
                // This is admittedly very dumb (e.g. it will mean we repeatedly
                // check the same prefix an absurd number of times), but its a
                // place to start.
                //
                // (One way to make this a little bit smarter would be to fold
                // in the parts generation code, and then figure out how to
                // "skip ahead" when we know a prefix of a given length doesn't
                // match and thus will *never* match.)
                let num_parts_iter = 1..=w.len();
                let indent = indent.clone();
                (Box::new(num_parts_iter.flat_map(move |num_parts| {
                    let env = env.clone();
                    let indent = indent.clone();
                    parts(w.len(), num_parts).flat_map(move |splits| {
                        println!("{}Kleene considering splits {:?}", indent, splits);
                        let all_r = std::iter::repeat(&**r);
                        self.split_matches(splits.into_iter(), env.clone(), w, all_r, accum+1).0
                    })
                })), accum)
            }
            RegularRightSide::EmptyLanguage => (None.b_iter(), accum),
        }
    }
}

#[cfg(test)]
mod tests;

#[test]
fn test_parts() {
    assert_eq!(parts(3, 1).collect::<Vec<_>>(),
               vec![vec![3]]);
    assert_eq!(parts(3, 2).collect::<Vec<_>>(),
               vec![vec![1,2], vec![2,1]]);
    assert_eq!(parts(4, 2).collect::<Vec<_>>(),
               vec![vec![1,3], vec![2,2], vec![3,1]]);
    assert_eq!(parts(4, 3).collect::<Vec<_>>(),
               vec![vec![1,1,2], vec![1,2,1],
                    vec![2,1,1]]);
    assert_eq!(parts(5, 3).collect::<Vec<_>>(),
               vec![vec![1,1,3], vec![1,2,2], vec![1,3,1],
                    vec![2,1,2], vec![2,2,1],
                    vec![3,1,1]]);
}

// given positive target T and positive count N where N <= T, returns iterator
// over length-N vectors of positive numbers that add up to T.
fn parts(target: usize, count: usize) -> impl Iterator<Item=Vec<usize>> {
    assert!(count <= target);

    // base case: trivial count
    if count == 1 {
        return vec![vec![target]].into_iter();
    }

    // recursive case: take a number, recur, map over results. repeat.
    let mut results: Vec<Vec<usize>> = Vec::new();
    for take in 1..=(target - count + 1) {
        let sub_target = target - take;
        let sub_count = count - 1;
        for sub_part in parts(sub_target, sub_count) {
            let mut took = vec![take];
            took.extend(sub_part.into_iter());
            results.push(took);
        }
    }
    return results.into_iter();
}

pub mod expr {
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum BinOp { Add, Sub, Mul, Div, Gt, Ge, Lt, Le, Eql, Neq }

    impl std::fmt::Display for BinOp {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(w, "{}", match *self {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Eql => "==",
                BinOp::Neq => "!=",
            })
        }
    }
    
    #[derive(PartialEq, Eq, Clone, Debug)]
    pub struct Var(pub String);

    // pub const Y_0: Var = Var("Y_0".into());
    pub fn y_0() -> Var { Var("Y_0".into()) }

    #[derive(PartialEq, Eq, Clone, Debug)]
    pub enum Expr { Var(Var), Lit(Val), BinOp(BinOp, Box<Expr>, Box<Expr>) }

    #[derive(PartialOrd, Ord, PartialEq, Eq, Clone, Debug)]
    pub enum Val { Bool(bool), Unit, String(String), Int(i64), }

    impl std::fmt::Display for Val {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Val::Bool(b) => write!(w, "{:?}", b),
                Val::Unit => write!(w, "()"),
                Val::String(s) => write!(w, "\"{}\"", s),
                Val::Int(i) => write!(w, "{:?}", i),
            }
        }
    }

    pub const TRUE: Val = Val::Bool(true);

    impl Expr {
        fn needs_parens(&self, ctxt: BinOp) -> bool {
            match (self, ctxt) {
                (Expr::Var(_), _) |
                (Expr::Lit(_), _) => false,

                (Expr::BinOp(inner_op, _lhs, _rhs), outer_op) => inner_op != &outer_op,
            }
        }
    }

    impl std::fmt::Display for Expr {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
           match self {
                Expr::Var(Var(v)) => write!(w, "{}", v),
                Expr::Lit(val) => write!(w, "{}", val),

                Expr::BinOp(op, lhs, rhs) => {
                    match (lhs.needs_parens(*op), rhs.needs_parens(*op)) {
                        (true, true) => 
                            write!(w, "({}) {} ({})", lhs, op, rhs),
                        (true, false) => 
                            write!(w, "({}) {} {}", lhs, op, rhs),
                        (false, true) => 
                            write!(w, "{} {} ({})", lhs, op, rhs),
                        (false, false) => 
                            write!(w, "{} {} {}", lhs, op, rhs),
                    }
                }
            }
        }
    }

    impl std::ops::Add<Val> for Val {
        type Output = Val;
        fn add(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs + rhs),
                (Val::String(mut lhs), Val::String(rhs)) => { lhs.push_str(&rhs); Val::String(lhs) }
                (lhs, rhs) => { panic!("invalid inputs for Add: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    impl std::ops::Sub<Val> for Val {
        type Output = Val;
        fn sub(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs - rhs),
                (lhs, rhs) => { panic!("invalid inputs for Sub: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    impl std::ops::Mul<Val> for Val {
        type Output = Val;
        fn mul(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs * rhs),
                (lhs, rhs) => { panic!("invalid inputs for Sub: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    impl std::ops::Div<Val> for Val {
        type Output = Val;
        fn div(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs / rhs),
                (lhs, rhs) => { panic!("invalid inputs for Sub: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct Env(Vec<(Var, Val)>);

    impl std::fmt::Display for Env {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
            let mut content: String = self.0.iter()
                .map(|(Var(x), val)| format!("{}={},", x, val))
                .collect();
            // drop last comma, if any content was added at all.
            content.pop();
            write!(w, "[{}]", content)
        }
    }

    impl Env {
        pub fn empty() -> Self { Env(vec![]) }

        pub fn bind(x: Var, v: Val) -> Self { Env(vec![(x, v)]) }

        pub fn extend(mut self, x: Var, v: Val) -> Self {
            self.0.retain(|(x_, _v_)| x_ != &x);
            self.0.push((x, v));
            self
        }

        pub fn lookup(&self, x: &Var) -> Option<&Val> {
            for (y, w) in self.0.iter().rev() {
                if x == y {
                    return Some(w);
                }
            }
            None
        }

        pub fn concat(self, e2: Env) -> Self {
            let mut s = self;
            for (x, v) in e2.0.into_iter() {
                s = s.extend(x, v);
            }
            s
        }
    }

    impl Expr {
        pub fn eval(&self, env: &Env) -> Val {
            match self {
                Expr::Var(x) => env.lookup(x).unwrap().clone(),
                Expr::Lit(v) => v.clone(),
                Expr::BinOp(op, e1, e2) => {
                    let lhs = e1.eval(env);
                    let rhs = e2.eval(env);
                    match op {
                        BinOp::Add => lhs + rhs,
                        BinOp::Sub => lhs - rhs,
                        BinOp::Mul => lhs * rhs,
                        BinOp::Div => lhs / rhs,
                        BinOp::Gt => Val::Bool(lhs > rhs),
                        BinOp::Ge => Val::Bool(lhs >= rhs),
                        BinOp::Lt => Val::Bool(lhs < rhs),
                        BinOp::Le => Val::Bool(lhs <= rhs),
                        BinOp::Eql => Val::Bool(lhs == rhs), 
                        BinOp::Neq => Val::Bool(lhs != rhs),
                   }
                }
            }
        }
    }

    impl From<char> for Var { fn from(c: char) -> Var { Var(c.to_string()) } }
    impl From<&str> for Var { fn from(c: &str) -> Var { Var(c.to_string()) } }
    impl From<String> for Var { fn from(c: String) -> Var { Var(c) } }
    
    impl From<Var> for Expr { fn from(x: Var) -> Expr { Expr::Var(x) } }
    impl From<Val> for Expr { fn from(v: Val) -> Expr { Expr::Lit(v) } }

    impl From<bool> for Val { fn from(b: bool) -> Val { Val::Bool(b) } }
    impl From<()> for Val { fn from((): ()) -> Val { Val::Unit } }
    impl From<String> for Val { fn from(s: String) -> Val { Val::String(s) } }
    impl From<&str> for Val { fn from(s: &str) -> Val { Val::String(s.to_string()) } }
    impl From<i64> for Val { fn from(n: i64) -> Val { Val::Int(n) } }
    impl From<&[super::Term]> for Val {
        fn from(terms: &[super::Term]) -> Val {
            use super::Term;
            let mut s = String::new();
            for t in terms {
                match t {
                    Term::C(c) => s.push(*c),
                    Term::S(s2) => s.push_str(&s2),
                }
            }
            Val::String(s)
        }
    }

    impl From<bool> for Expr { fn from(b: bool) -> Expr { let v: Val = b.into(); v.into() } }
    impl From<()> for Expr { fn from((): ()) -> Expr { let v: Val = ().into(); v.into() } }
    impl From<String> for Expr { fn from(s: String) -> Expr { let v: Val = s.into(); v.into() } }
    impl From<&str> for Expr { fn from(s: &str) -> Expr { let v: Val = s.into(); v.into() } }

}

lalrpop_mod!(pub yakker); // synthesized by LALRPOP

#[derive(PartialEq, Debug)]
pub enum YakkerError {
    NoCharAfterBackslash,
    UnrecognizedChar(char),
}

fn normalize_escapes(input: &str) -> Result<String, YakkerError> {
    let mut s = String::with_capacity(input.len());
    let mut cs = input.chars();
    while let Some(c) = cs.next() {
        if c == '\\' {
            match cs.next() {
                None => return Err(YakkerError::NoCharAfterBackslash),
                Some(c @ '\\') | Some(c @ '"') => { s.push(c); continue }
                Some('n') => { s.push('\n'); continue }
                Some('t') => { s.push('\t'); continue }
                Some('r') => { s.push('\r'); continue }
                Some(c) => return Err(YakkerError::UnrecognizedChar(c)),
           }
        } else {
            s.push(c);
        }
    }
    return Ok(s);
}

// A grammar G is a tuple (Sigma, Delta, Phi, A_0, R), where
//   Sigma is a finite set of terminals
//   Delta is a finite set of non-terminals
//   Phi si a finite set of blackboxes
//   A_0 in Delta is the start non-terminal, and
//   R maps non-termainsl to regular right sides

// Regular right sides
//
// r ::= epsilon           <empty string>
//    |  empty             <empty language>
//    |  c                 <terminal>
//    |  x := A(e)         <nonterminal>
//    |  x := e            <binding>
//    |  (r r)             <concatenation>
//    |  (r | r)           <alternation>
//    |  (r*)              <Kleene closure>
//    |  [e]               <constraint>
//    |  phi(e)            <blackbox>
//

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub enum Term { C(char), S(String) }

impl Term {
    fn string(&self) -> String {
        let mut s = String::new();
        match self {
            Term::C(c) => { s.push(*c) }
            Term::S(s2) => { s = s2.clone(); }
        }
        s
    }
    fn matches(&self, w: &[Term]) -> bool {
        if let (&Term::C(c1), &[Term::C(c2)]) = (self, w) {
            return c1 == c2;
        }
        let left = self.string();
        let left = left.chars().fuse();
        let right: Vec<String> = w.iter().map(|t|t.string()).collect();
        let right = right.iter().map(|s|s.chars()).flatten();
        left.eq(right)
    }
}

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub struct NonTerm(String);
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct Var(String);
#[derive(PartialEq, Eq, Clone)]
pub struct Val(String);

// notation from paper: `{x := v }`
#[derive(PartialEq, Eq, Clone)]
pub struct Binding(expr::Var, Val);

// notation from paper: `< w >`
#[derive(PartialEq, Eq, Clone)]
pub struct BlackBox(String);

impl From<&str> for Term { fn from(a: &str) -> Self { Self::S(a.into()) } }
impl From<&str> for NonTerm { fn from(a: &str) -> Self { Self(a.into()) } }
impl From<&str> for Val { fn from(v: &str) -> Self { Self(v.into()) } }

// notation from paper: `x:A(v)XXX`,
// e.g. `x:A(v)< T' >`
// e.g. `x:A(v) := w`

pub struct Parsed<X> { var: expr::Var, nonterm: NonTerm, input: Val, payload: X }

pub enum AbstractNode<X> {
    Term(Term),
    Binding(Binding),
    BlackBox(BlackBox),
    Parse(Parsed<X>),
}

pub struct Tree(pub Vec<AbstractNode<Tree>>);

// W : AbstractString
// m : AbstractString that is nonterminal-free
pub struct AbstractString(Vec<AbstractNode<String>>);

impl AbstractString {
    // detetermines if W can be treated as an m.
    pub fn nonterminal_free(&self) -> bool {
        for n in &self.0 {
            if let AbstractNode::Parse(..) = n {
                return false;
            }
        }
        return true;
    }

    // ||W|| from paper
    pub fn erased(&self) -> String {
        let mut accum = String::new();
        for n in &self.0 {
            let backing: String;
            let s: &str = match n {
                AbstractNode::Term(Term::S(s)) => s,
                AbstractNode::Term(Term::C(c)) => { backing = [c].into_iter().collect(); &backing }
                AbstractNode::Binding(_) => continue,
                AbstractNode::BlackBox(bb) => &bb.0,
                AbstractNode::Parse(p) => &p.payload,
            };
            accum.push_str(s);
        }
        accum
    }

    // Strings(W, A, v) from paper
    pub fn strings(&self, a: NonTerm, v: Val) -> Vec<&str> {
        let mut accum: Vec<&str> = Vec::new();
        for n in &self.0 {
            if let AbstractNode::Parse(p) = n {
                if p.nonterm == a && p.input == v {
                    accum.push(&p.payload);
                }
            }
        }
        accum
    }
}

impl Tree {
    pub fn leaves<'a>(&'a self) -> Vec<std::borrow::Cow<'a, str>> {
        use std::borrow::Cow;
        let mut accum: Vec<Cow<str>> = Vec::new();
        for n in &self.0  {
            match n {
                AbstractNode::Term(Term::S(s)) => accum.push(s.into()),
                AbstractNode::Term(Term::C(c)) => {
                    let mut s = String::new();
                    s.push(*c);
                    accum.push(s.into());
                }
                AbstractNode::Binding(_) => continue,
                AbstractNode::BlackBox(bb) => accum.push((&bb.0).into()),
                AbstractNode::Parse(p) => {
                    accum.extend(p.payload.leaves().into_iter())
                }
            }
        }
        accum
    }
    pub fn roots(&self) -> AbstractString {
        let mut accum: Vec<AbstractNode<String>>= Vec::new();
        for n in &self.0  {
            match n {
                AbstractNode::Term(t) => accum.push(AbstractNode::Term(t.clone())),
                AbstractNode::Binding(b) => accum.push(AbstractNode::Binding(b.clone())),
                AbstractNode::BlackBox(bb) => accum.push(AbstractNode::BlackBox(bb.clone())),
                AbstractNode::Parse(Parsed{ var, nonterm, input, payload }) => {
                    let var = var.clone();
                    let nonterm = nonterm.clone();
                    let input = input.clone();
                    let mut leaves = String::new();
                    for leaf in payload.leaves() {
                        leaves.push_str(&leaf);
                    }
                    let payload = leaves;
                    accum.push(AbstractNode::Parse(Parsed {
                        var, nonterm, input, payload }));
                }
            }
        }
        AbstractString(accum)
    }
}
    
