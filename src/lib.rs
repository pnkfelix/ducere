#[macro_use] extern crate lalrpop_util;

pub trait Recognizer {
    type Term;
    type String;
    fn accept(&self, iter: &mut dyn Iterator<Item=&Self::Term>) -> Option<Self::String>;
}

pub struct Blackbox {
    name: String,
    from_val: Box<dyn Fn(expr::Val) -> Box<dyn Recognizer<Term=Term, String=String>>>,
}

impl PartialEq for Blackbox {
    fn eq(&self, other: &Self) -> bool {
        ((&*self.from_val) as *const _) == ((&*other.from_val) as *const _)
    }
}

impl Eq for Blackbox { }

impl std::fmt::Debug for Blackbox {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, "blackbox[{}]", self.name)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Grammar { pub rules: Vec<Rule> }

impl Grammar {
    pub fn empty() -> Self { Grammar { rules: vec![] } }

    fn rule(&self, nonterm: &NonTerm) -> Option<&Rule> {
        self.rules.iter().find(|r| &r.0 == nonterm)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Rule(NonTerm, Option<expr::Var>, RegularRightSide);

#[derive(PartialEq, Eq, Debug)]
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
    pub fn matches(&self, env: &expr::Env, w: &[Term], r: &RegularRightSide) -> Option<expr::Env> {
        match r {
            // GL-EPS
            RegularRightSide::EmptyString =>
                if w.len() == 0 {
                    Some(expr::Env::empty())
                } else {
                    None
                },
            // GL-TERM
            RegularRightSide::Term(t) =>
                if t.matches(w) {
                    Some(expr::Env::empty())
                } else {
                    None
                },
            // GL-PRED
            RegularRightSide::Constraint(e) =>
                if e.eval(env) == expr::TRUE {
                    Some(expr::Env::empty())
                } else {
                    None
                }
            // GL-BIND
            RegularRightSide::Binding { x, e } =>
                Some(expr::Env::bind(x.clone(), e.eval(env))),
            // GL-φ
            RegularRightSide::Blackbox(bb, e) => {
                let v = e.eval(env);
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
                    Some(expr::Env::empty())
                } else {
                    None
                }
            }
            // GL-A
            RegularRightSide::NonTerm { x, A, e } => {
                let Rule(_a, opt_var, subrule) = self.rule(A).unwrap();
                let subenv = match (e, opt_var) {
                    (None, None) =>
                        // great: non-parameterized terminals don't need values
                        expr::Env::empty(),
                    (Some(e), Some(y_0)) => {
                        let v = e.eval(env);
                        expr::Env::bind(y_0.clone(), v)
                    }
                    (Some(e), None) => {
                        // not as great: I'd prefer to not use y_0 formalism.
                        let v = e.eval(env);
                        expr::Env::bind(expr::y_0(), v)
                    }
                    (None, Some(y_0)) => {
                        panic!("provided expr argument {:?} \
                                to *unparameterized* non-term {:?}({:?})",
                               e, _a, y_0);
                    }
                };
                let subresult = self.matches(&subenv, w, subrule);
                subresult.map(|_discarded_env| {
                    match x {
                        Some(x) => expr::Env::bind(x.clone(), w.into()),
                        None => expr::Env::empty(),
                    }
                })
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
                    println!("`{:?}` in Concat({:?},{:?}) trial i={}",
                             w, r1, r2, i);
                    let (w1,  w2) = w.split_at(i);
                    let w1_in_r1 = self.matches(env, w1, r1);
                    println!("w1_in_r1: {:?}", w1_in_r1);
                    let env1 = if let Some(env1) = w1_in_r1 {
                        env1
                    } else {
                        continue;
                    };
                    let new_env = env.clone().concat(env1.clone());
                    let w2_in_r2 = self.matches(&new_env, w2, r2);
                    println!("w2_in_r2: {:?}", w2_in_r2);
                    let env2 = if let Some(env2) = w2_in_r2 {
                        env2
                    } else {
                        continue;
                    };
                    return Some(env1.concat(env2));
                }
                return None;
            }
            // GL-ALTL + GL-ALTR
            RegularRightSide::Either(r1, r2) => {
                let result1 = self.matches(env, w, r1);
                if result1.is_some() {
                    return result1;
                }
                let result2 = self.matches(env, w, r2);
                if result2.is_some() {
                    return result2;
                }
                return None;
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
            RegularRightSide::Kleene(r) => {
                // if the string is empty, we trivially match
                if w.len() == 0 {
                    return Some(expr::Env::empty());
                }
                // likewise, if the rule can match the whole string, then do that too
                if let Some(e1) = self.matches(env, w, r) {
                    return Some(e1);
                }
                // otherwise, we will enumerate the ways to break up the string.
                // This is admittedly very dumb (e.g. it will mean we repeatedly
                // check the same prefix an absurd number of times), but its a
                // place to start.
                for num_parts in 2..=w.len() {
                    'next_splits: for splits in parts(w.len(), num_parts) {
                        let mut w_suffix = w;
                        let mut accum_eval_env = env.clone();
                        let mut accum_ret_env = expr::Env::empty();
                        for split in splits {
                            let (w_pre, w_post) = w_suffix.split_at(split);
                            if let Some(e_pre) = self.matches(&accum_eval_env, w_pre, r) {
                                w_suffix = w_post;
                                accum_eval_env = accum_eval_env.concat(e_pre.clone());
                                accum_ret_env = accum_ret_env.concat(e_pre);
                            } else {
                                continue 'next_splits;
                            }
                        }
                        assert_eq!(w_suffix.len(), 0);
                        return Some(accum_ret_env);
                    }
                }
                return None;
            }
            RegularRightSide::EmptyLanguage => None,
        }
    }
}

#[test]
fn regular_right_sides() {
    let g = Grammar::empty();
    let emp = &expr::Env::empty();
    fn right_side(s: &str) -> RegularRightSide {
        yakker::RegularRightSideParser::new().parse(s).unwrap()
    }
    fn input(s: &str) -> Vec<Term> {
        s.chars().map(|c| Term::C(c)).collect()
    }
    assert!(g.matches(emp, &input("c"), &right_side(r"'c'")).is_some());
    assert!(g.matches(emp, &input("d"), &right_side(r"'c'")).is_none());
    assert!(g.matches(emp, &input("ab"), &right_side(r"'a''b'")).is_some());
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
    for take in 1..target {
        let sub_target = target - take;
        let sub_count = count - 1;
        // (TODO: change upper-bound to eliminate need to do this case.)
        if sub_count > sub_target { continue; }
        for sub_part in parts(sub_target, sub_count) {
            let mut took = vec![take];
            took.extend(sub_part.into_iter());
            results.push(took);
        }
    }
    return results.into_iter();
}

pub mod expr {
    #[derive(PartialEq, Eq, Clone, Debug)]
    pub enum BinOp { Add, Sub, Mul, Div, Gt, Ge, Lt, Le, Eql, Neq }
    
    #[derive(PartialEq, Eq, Clone, Debug)]
    pub struct Var(pub String);

    // pub const Y_0: Var = Var("Y_0".into());
    pub fn y_0() -> Var { Var("Y_0".into()) }

    #[derive(PartialEq, Eq, Clone, Debug)]
    pub enum Expr { Var(Var), Lit(Val), BinOp(BinOp, Box<Expr>, Box<Expr>) }

    #[derive(PartialOrd, Ord, PartialEq, Eq, Clone, Debug)]
    pub enum Val { Bool(bool), Unit, String(String), Int(i64), }

    pub const TRUE: Val = Val::Bool(true);

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

    impl Env {
        pub fn empty() -> Self { Env(vec![]) }

        pub fn bind(x: Var, v: Val) -> Self { Env(vec![(x, v)]) }

        pub fn extend(mut self, x: Var, v: Val) -> Self {
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

        pub fn concat(mut self, e2: Env) -> Self {
            self.0.extend(e2.0.into_iter());
            self
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

#[test]
fn yakker() {
    use expr::{Expr};

    assert_eq!(yakker::VarParser::new().parse("x"), Ok('x'.into()));

    assert_eq!(yakker::ExprParser::new().parse("x"), Ok(Expr::Var('x'.into())));
    assert_eq!(yakker::ExprParser::new().parse("x"), Ok(Expr::Var('x'.into())));
    assert_eq!(yakker::ExprParser::new().parse("true"), Ok(true.into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""..""#), Ok("..".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""xx""#), Ok("xx".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""x""#), Ok("x".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""""#), Ok("".into()));

    assert_eq!(yakker::ExprParser::new().parse(r#""\"""#), Ok("\"".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""\n""#), Ok("\n".into()));

    assert_eq!(yakker::RegularRightSideParser::new().parse(r"'c'"), Ok(RegularRightSide::Term("c".into())));
    assert_eq!(yakker::RegularRightSideParser::new().parse(r"'c''d'"),
               Ok(RegularRightSide::Concat(Box::new(RegularRightSide::Term("c".into())),
                                           Box::new(RegularRightSide::Term("d".into()))
               )));
    assert_eq!(yakker::NonTermParser::new().parse(r"A"), Ok("A".into()));

    assert_eq!(yakker::RuleParser::new().parse(r"A::='c'"), Ok(Rule("A".into(), None, RegularRightSide::Term("c".into()))));

    assert_eq!(yakker::GrammarParser::new().parse(r"A::='c'"), Ok(Grammar { rules: vec![Rule("A".into(), None, RegularRightSide::Term("c".into()))]}));
    assert_eq!(yakker::GrammarParser::new().parse(r"A::='a' B::='b'"), Ok(Grammar { rules: vec![Rule("A".into(), None, RegularRightSide::Term("a".into())),
    Rule("B".into(), None, RegularRightSide::Term("b".into()))]}));
}

// Example: Imperative fixed-width integer
//
// int(n) = ([n > 0]( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' ) { n:=n-1 })* [n = 0]

#[cfg(test)]
macro_rules! assert_matches {
    ($e:expr, $p:pat) => {
        let v = $e;
        if let $p = v { } else {
            panic!("assert fail {:?} does not match pattern {}", v, stringify!($p));
        }
    }
}

#[test]
fn imperative_fixed_width_integer() {
    assert_matches!(yakker::NonTermParser::new().parse("Int"), Ok(_));
    assert_matches!(yakker::RegularRightSideParser::new().parse("x:=Int(())"), Ok(_));
    assert_matches!(yakker::RightSideLeafParser::new().parse("'0'"), Ok(_));
    assert_matches!(yakker::RuleParser::new().parse("Int ::= '0' "), Ok(_));
    assert_matches!(yakker::RuleParser::new().parse("Int ::= ( ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') )* "), Ok(_));

    assert_matches!(yakker::RuleParser::new().parse("Int ::= { n:=y_0 } ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0]"), Ok(_));
    assert_matches!(yakker::RuleParser::new().parse("Int(n) ::= ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0]"), Ok(_));
}

// Example: Functional fixed-width integer
//
//    dig = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
// int(n) = [n = 0] | [n > 0] dig int(n - 1)

// Example: Left-factoring
//
// A = (B '?') | (C '!')
// B = 'x' + 'x'
// C = ('x' + 'x') | ('x' - 'x')


// Example: "Regular right sides"
//
//  digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
// number = digit digit*

// Example: "Full context-free grammars"
//
// typename = identifer
//     expr = '&' expr               ; address of
//          | expr '&' expr          ; bitwise conjunction
//          | '(' typename ')' expr  ; cast
//          | '(' expr ')'
//          | identifier
//          | ...

// Example: "Attribute-directed parsing"
// literal8 = '~' '{' x:=number ('+' | epsilon) '}' { n := string2int(x) } CRLF ([n > 0] OCTET { n := n - 1 })* [n = 0]

// Example: "Parmeterized Nonterminals"
//
// stringFW(n) = ([n > 0] CHAR8 { n := n-1 })* [n = 0]
//
// stringFW(n) = [n = 0] | [n > 0] CHAR8 stringFW(n - 1)
//
//    decls(s) = 'typedef' type-expr x := identifier decls(insert(s.types,x))
//             | ...
// 
// typename(s) = x:=identifier [ member(s.types,x) ]
//     expr(s) = '&' expr(s)                              ; address of
//             | expr(s) '&' expr(s)                      ; bitwise conjunction
//             | '{' typename(s) '}' expr(s)              ; cast
//             | '(' expr(s) ')'
//             | x:=identifier [ not(member(s.types,x)) ]

// Example: "Scannerless parsing"
//
// statement = 'return' (SP | TAB)* (LF | ';' | identifier ';')

// Example: "Blackbox parsers"
// *blackbox* bbdate
// entry = client SP auth-id SP auth-id SP '[' bbdate ']' SP request SP response SP length

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

#[derive(PartialEq, Eq, Clone, Debug)]
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
        let left = self.string();
        let left = left.chars().fuse();
        let right: Vec<String> = w.iter().map(|t|t.string()).collect();
        let right = right.iter().map(|s|s.chars()).flatten();
        left.eq(right)
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
