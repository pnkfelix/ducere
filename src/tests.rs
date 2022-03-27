use super::*;

pub(crate) fn right_side(s: &str) -> RegularRightSide {
    yakker::RegularRightSideParser::new().parse(s).unwrap()
}
pub(crate) fn input(s: &str) -> Vec<Term> {
    s.chars().map(|c| Term::C(c)).collect()
}

#[test]
fn regular_right_sides() {
    let g = Grammar::empty();
    assert!(g.matches(&input("c"), &right_side(r"'c'")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"'c'")).no_parse());
    assert!(g.matches(&input("ab"), &right_side(r"'a''b'")).has_parse());
    assert!(g.matches(&input("ac"), &right_side(r"'a''b'")).no_parse());
    assert!(g.matches(&input("abc"), &right_side(r"'a''b''c'")).has_parse());
    assert!(g.matches(&input("a"), &right_side(r"'a'|'b'")).has_parse());
    assert!(g.matches(&input("b"), &right_side(r"'a'|'b'")).has_parse());
    assert!(g.matches(&input("aaa"), &right_side(r"'a'*")).has_parse());
    assert!(g.matches(&input("aaa"), &right_side(r"('a')*")).has_parse());
    assert!(g.matches(&input("aaa"), &right_side(r"('a'|'b')*")).has_parse());
    assert!(g.matches(&input("aba"), &right_side(r"('a'|'b')*")).has_parse());
    assert!(g.matches(&input("aca"), &right_side(r"('a'|'b')*")).no_parse());
}

#[test]
fn regular_right_sides_expression_dsl() {
    let g = Grammar::empty();
    assert!(g.matches(&input(""), &right_side(r#"{x:=3}"#)).has_parse());
    assert!(g.matches(&input(""), &right_side(r#"{x:=3}[x==3]"#)).has_parse());
    assert!(g.matches(&input(""), &right_side(r#"{x:=3}[x==4]"#)).no_parse());
    assert!(g.matches(&input(""), &right_side(r#"{x:=3}{x:=4}[x==4]"#)).has_parse());
}

#[test]
fn non_empty_grammar() {
    let g1 = yakker::GrammarParser::new().parse(r"A::='c'").unwrap();
    assert!(g1.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g1.matches(&input("d"), &right_side(r"<x:=A(0)>")).no_parse());
    let g2 = yakker::GrammarParser::new().parse(r"B::='d'").unwrap();
    assert!(g2.matches(&input("c"), &right_side(r"<x:=B(0)>")).no_parse());
    assert!(g2.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
    let g3 = Grammar { rules: g1.rules.into_iter().chain(g2.rules.into_iter()).collect() };
    assert!(g3.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g3.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
    let g4 = yakker::GrammarParser::new().parse(r"A::='c'; B::='d'").unwrap();
    assert!(g4.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g4.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
    let g5 = yakker::GrammarParser::new().parse(r"A::=<x:=C(0)>; B::=<x:=D(1)>; C::='c'; D::='d';").unwrap();
    assert!(g5.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g5.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
}

#[test]
fn grammar_sugar() {
    let g = yakker::GrammarParser::new().parse(r"A::=<Z(0)>; B::=<y:=D>; Z::=<C>; C::='c'; D::='d'").unwrap();
    assert!(g.matches(&input("c"), &right_side(r"<x:=A>")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"<x:=A>")).no_parse());
    assert!(g.matches(&input("d"), &right_side(r"<B>")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"<A>")).no_parse());
    assert!(g.matches(&input("c"), &right_side(r"x:=A")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"x:=A")).no_parse());
    assert!(g.matches(&input("d"), &right_side(r"B")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"A")).no_parse());
    let g = yakker::GrammarParser::new().parse(r"A::=<Z(0)>; B::=y:=D; Z::=C; C::='c'; D::='d'").unwrap();
    assert!(g.matches(&input("c"), &right_side(r"<x:=A>")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"<x:=A>")).no_parse());
    assert!(g.matches(&input("d"), &right_side(r"<B>")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"<A>")).no_parse());
    assert!(g.matches(&input("c"), &right_side(r"x:=A")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"x:=A")).no_parse());
    assert!(g.matches(&input("d"), &right_side(r"B")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"A")).no_parse());
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
    assert_eq!(yakker::GrammarParser::new().parse(r"A::='a'; B::='b'"), Ok(Grammar { rules: vec![Rule("A".into(), None, RegularRightSide::Term("a".into())),
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
fn imperative_fixed_width_integer_foundations() {
    assert_matches!(yakker::NonTermParser::new().parse("Int"), Ok(_));
    assert_matches!(yakker::RegularRightSideParser::new().parse("<x:=Int(())>"), Ok(_));
    assert_matches!(yakker::RightSideLeafParser::new().parse("'0'"), Ok(_));
    assert_matches!(yakker::RuleParser::new().parse("Int ::= '0' "), Ok(_));
    assert_matches!(yakker::RuleParser::new().parse("Int ::= ( ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') )* "), Ok(_));

    assert_matches!(yakker::RuleParser::new().parse("Int ::= { n:=y_0 } ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0]"), Ok(_));
    assert_matches!(yakker::RuleParser::new().parse("Int(n) ::= ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0]"), Ok(_));
}

#[cfg(test)]
fn imperative_fixed_width_integer_grammar() -> Grammar {
    yakker::GrammarParser::new().parse(r"Int(n) ::= ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0];").unwrap()
}

#[test]
fn imperative_fixed_width_integer_1() {
    use tests::{input, right_side};

    let g = imperative_fixed_width_integer_grammar();
    assert!(g.matches(&input("1"), &right_side(r"<Int(1)>")).has_parse());
    assert!(g.matches(&input("0"), &right_side(r"<Int(1)>")).has_parse());
}

#[test]
fn simpler_variant_on_ifwi2_a() {
    use tests::{input, right_side};

    assert!(yakker::GrammarParser::new().parse(r"S(n) ::= [n gt 0] 'a' { n := n-1 } [n gt 0] 'b' { n := n-1 } [n eql 0];")
            .unwrap()
            .matches(&input("ab"), &right_side(r"<S(2)>"))
            .has_parse());
}

#[test]
fn simpler_variant_on_ifwi2_b() {
    use tests::{input, right_side};

    assert!(yakker::GrammarParser::new().parse(r"S(n) ::= ([n gt 0] ( 'a' | 'b' ) { n := n- 1 })* [n eql 0];")
            .unwrap()
            .matches(&input("ab"), &right_side(r"<S(2)>"))
            .has_parse());
}

#[test]
fn imperative_fixed_width_integer_2() {
    use tests::{input, right_side};
    let g = imperative_fixed_width_integer_grammar();
    assert!(g.matches(&input("10"), &right_side(r"<Int(2)>")).has_parse());
}

// Example: Functional fixed-width integer
//
//    dig = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
// int(n) = [n = 0] | [n > 0] dig int(n - 1)

#[test]
fn functional_fixed_width_integer() {

}

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

