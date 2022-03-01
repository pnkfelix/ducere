#[macro_use] extern crate lalrpop_util;

pub trait Recognizer {
    type Term;
    type String;
    fn accept(&self, iter: &dyn Iterator<Item=Self::Term>) -> Option<Self::String>;
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
pub struct Rule(NonTerm, RegularRightSide);

#[derive(PartialEq, Eq, Debug)]
pub enum RegularRightSide {
    EmptyString,
    EmptyLanguage,
    Term(Term),
    #[allow(non_snake_case)]
    NonTerm { x: expr::Var, A: NonTerm, e: expr::Expr },
    Binding { x: expr::Var, e: expr::Expr },
    Concat(Box<Self>, Box<Self>),
    Either(Box<Self>, Box<Self>),
    Kleene(Box<Self>),
    Constraint(expr::Expr),
    Blackbox(Blackbox),

}

pub mod expr {
    #[derive(PartialEq, Eq, Clone, Debug)]
    pub struct Var(pub char);

    #[derive(PartialEq, Eq, Clone, Debug)]
    pub enum Expr { Var(Var), Lit(Val) }

    #[derive(PartialEq, Eq, Clone, Debug)]
    pub enum Val { Bool(bool), Unit, String(String) }

    #[derive(Clone, Debug)]
    pub struct Env(Vec<(Var, Val)>);

    impl Env {
        pub fn empty() -> Self { Env(vec![]) }

        pub fn bind(mut self, x: Var, v: Val) -> Self {
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
        pub fn eval(&self, env: Env) -> Val {
            match self {
                Expr::Var(x) => env.lookup(x).unwrap().clone(),
                Expr::Lit(v) => v.clone(),
            }
        }
    }

    impl From<Var> for Expr { fn from(x: Var) -> Expr { Expr::Var(x) } }
    impl From<Val> for Expr { fn from(v: Val) -> Expr { Expr::Lit(v) } }

    impl From<bool> for Val { fn from(b: bool) -> Val { Val::Bool(b) } }
    impl From<()> for Val { fn from((): ()) -> Val { Val::Unit } }
    impl From<String> for Val { fn from(s: String) -> Val { Val::String(s) } }
    impl From<&str> for Val { fn from(s: &str) -> Val { Val::String(s.to_string()) } }

    impl From<bool> for Expr { fn from(b: bool) -> Expr { let v: Val = b.into(); v.into() } }
    impl From<()> for Expr { fn from((): ()) -> Expr { let v: Val = ().into(); v.into() } }
    impl From<String> for Expr { fn from(s: String) -> Expr { let v: Val = s.into(); v.into() } }
    impl From<&str> for Expr { fn from(s: &str) -> Expr { let v: Val = s.into(); v.into() } }

}

lalrpop_mod!(pub yakker); // synthesized by LALRPOP

#[test]
fn yakker() {
    use expr::{Expr, Var};

    assert_eq!(yakker::VarParser::new().parse("x"), Ok(Var('x')));

    assert_eq!(yakker::ExprParser::new().parse("x"), Ok(Expr::Var(Var('x'))));
    assert_eq!(yakker::ExprParser::new().parse("x"), Ok(Var('x').into()));
    assert_eq!(yakker::ExprParser::new().parse("true"), Ok(true.into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""..""#), Ok("..".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""xx""#), Ok("xx".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""x""#), Ok("x".into()));
    assert_eq!(yakker::ExprParser::new().parse(r#""""#), Ok("".into()));

    // XXX for now, I'm going to skip handling escape sequences in strings.
    // Instead, it will "just" be illegal to have delimiter characters in string literals.
    // assert_eq!(yakker::ExprParser::new().parse(r#""\"""#), Ok("\"".into()));

    assert_eq!(yakker::RegularRightSideParser::new().parse(r"c"), Ok(RegularRightSide::Term("c".into())));
    assert_eq!(yakker::NonTermParser::new().parse(r"A"), Ok("A".into()));

    assert_eq!(yakker::RuleParser::new().parse(r"A::=c"), Ok(Rule("A".into(), RegularRightSide::Term("c".into()))));
}

// Example: Imperative fixed-width integer
//
// int(n) = ([n > 0]( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' ) { n:=n-1 })* [n = 0]

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
pub struct Term(String);
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct NonTerm(String);
#[derive(PartialEq, Eq, Clone)]
pub struct Var(String);
#[derive(PartialEq, Eq, Clone)]
pub struct Val(String);

// notation from paper: `{x := v }`
#[derive(PartialEq, Eq, Clone)]
pub struct Binding(Var, Val);

// notation from paper: `< w >`
#[derive(PartialEq, Eq, Clone)]
pub struct BlackBox(String);

impl From<&str> for Term { fn from(a: &str) -> Self { Self(a.into()) } }
impl From<&str> for NonTerm { fn from(a: &str) -> Self { Self(a.into()) } }
impl From<&str> for Val { fn from(v: &str) -> Self { Self(v.into()) } }

// notation from paper: `x:A(v)XXX`,
// e.g. `x:A(v)< T' >`
// e.g. `x:A(v) := w`

pub struct Parsed<X> { var: Var, nonterm: NonTerm, input: Val, payload: X }

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
            let s: &str = match n {
                AbstractNode::Term(t) => &t.0,
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
    pub fn leaves(&self) -> Vec<&str> {
        let mut accum: Vec<&str> = Vec::new();
        for n in &self.0  {
            match n {
                AbstractNode::Term(t) => accum.push(&t.0),
                AbstractNode::Binding(_) => continue,
                AbstractNode::BlackBox(bb) => accum.push(&bb.0),
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
                        leaves.push_str(leaf);
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
