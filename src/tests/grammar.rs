use crate::*;
use super::*;
use crate::specification_rules::grammar::GrammarInterpreter;

pub(crate) fn right_side(s: &str) -> RegularRightSide {
    let lex = toyman::Lexer::new(s);
    yakker::RegularRightSideParser::new().parse(s, lex).unwrap()
}

#[test]
fn regular_right_sides_single() {
    let g = Grammar::empty();
    assert!(g.matches(&input("c"), &right_side(r"'c'")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"'c'")).no_parse());
}

#[test]
fn regular_right_sides() {
    let g = Grammar::empty();
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
fn regular_right_sides_expression_dsl_as_non_term_actual_arg() {
    let s = r#"<StringFW(3)>"#; let lex = toyman::Lexer::new(s); yakker::RegularRightSideParser::new().parse(s, lex).unwrap();
    let s = r#"<StringFW(n - 1)>"#; let lex = toyman::Lexer::new(s); yakker::RegularRightSideParser::new().parse(s, lex).unwrap();
}

#[test]
fn non_empty_grammar() {
    let s = r"A::='c'";
    let lex = toyman::Lexer::new(s);
    let g1 = yakker::GrammarParser::new().parse(s, lex).unwrap();
    assert!(g1.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g1.matches(&input("d"), &right_side(r"<x:=A(0)>")).no_parse());
    let s = r"B::='d'";
    let lex = toyman::Lexer::new(s);
    let g2 = yakker::GrammarParser::new().parse(s, lex).unwrap();
    assert!(g2.matches(&input("c"), &right_side(r"<x:=B(0)>")).no_parse());
    assert!(g2.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
    let g3 = Grammar { rules: g1.rules.into_iter().chain(g2.rules.into_iter()).collect() };
    assert!(g3.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g3.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
    let s = r"A::='c';B::='d'";
    let lex = toyman::Lexer::new(s);
    let g4 = yakker::GrammarParser::new().parse(s, lex).unwrap();
    assert!(g4.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g4.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
    let s = r"A::= <x:=C(0)>;B::= <x:=D(1)>;C::='c';D::='d';";
    let lex = toyman::Lexer::new(s);
    let g5 = yakker::GrammarParser::new().parse(s, lex).unwrap();
    assert!(g5.matches(&input("c"), &right_side(r"<x:=A(0)>")).has_parse());
    assert!(g5.matches(&input("d"), &right_side(r"<x:=B(0)>")).has_parse());
}

#[test]
fn grammar_sugar() {
    let s = r"A::= <Z(0)>;B::= <y:=D>;Z::= <C>;C::='c';D::='d'";
    let lex = toyman::Lexer::new(s);
    let g = yakker::GrammarParser::new().parse(s, lex).unwrap();
    assert!(g.matches(&input("c"), &right_side(r"<x:=A>")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"<x:=A>")).no_parse());
    assert!(g.matches(&input("d"), &right_side(r"<B>")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"<A>")).no_parse());
    assert!(g.matches(&input("c"), &right_side(r"x:=A")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"x:=A")).no_parse());
    assert!(g.matches(&input("d"), &right_side(r"B")).has_parse());
    assert!(g.matches(&input("d"), &right_side(r"A")).no_parse());
    let s = r"A::= <Z(0)>;B::=y:=D;Z::=C;C::='c';D::='d'";
    let lex = toyman::Lexer::new(s);
    let g = yakker::GrammarParser::new().parse(s, lex).unwrap();
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
fn grammar_with_multi_char_nonterms() {
    let s = r"Aone ::= 'a' '1'; Atwo ::= 'a' '2'";
    let lex = toyman::Lexer::new(s);
    let g = yakker::GrammarParser::new().parse(s, lex).unwrap();

    // FIXME: we cannot test this because it causes code, with the simple fix, to panic.
    // But the idea is there.
    // assert!(g.matches(&input("a1"), &right_side(r"A")).no_parse());

    assert!(g.matches(&input("a1"), &right_side(r"Aone")).has_parse());
    assert!(g.matches(&input("a2"), &right_side(r"Atwo")).has_parse());
    assert!(g.matches(&input("a1"), &right_side(r"Atwo")).no_parse());
    assert!(g.matches(&input("a2"), &right_side(r"Aone")).no_parse());
}

macro_rules! parse_from {
    ($KindParser:ident $s:expr) => {
        {
            let s = $s;
            let lex = toyman::Lexer::new(s);
            yakker::$KindParser::new().parse(s, lex)
        }
    }
}

#[test]
fn yakker_ident() {
    use expr::{Expr};

    assert_eq!(parse_from!(VarParser "x"), Ok('x'.into()));
    assert_eq!(parse_from!(ExprParser "x"), Ok(Expr::Var('x'.into())));
    assert_eq!(parse_from!(ExprParser "x"), Ok(Expr::Var('x'.into())));

    assert_eq!(parse_from!(ExprParser "true"), Ok(true.into()));
}

#[test]
fn yakker_basic_double_quotes() {
    {
        let s = r#""..""#;
        let lex = toyman::Lexer::new(s);
        assert_eq!(yakker::MulArgParser::new().parse(s, lex), Ok("..".into()));
    }

    assert_eq!(parse_from!(MulArgParser r#""..""#), Ok("..".into()));
    assert_eq!(parse_from!(ExprParser r#""..""#), Ok("..".into()));
    assert_eq!(parse_from!(ExprParser r#""xx""#), Ok("xx".into()));
    assert_eq!(parse_from!(ExprParser r#""x""#), Ok("x".into()));
    assert_eq!(parse_from!(ExprParser r#""""#), Ok("".into()));
    assert_eq!(parse_from!(ExprParser r#""(""#), Ok("(".into()));
    assert_eq!(parse_from!(ExprParser r#"")""#), Ok(")".into()));
}
#[test]
fn yakker_basic_double_quotes_escapes() {
    assert_eq!(parse_from!(ExprParser r#""\"""#), Ok("\"".into()));
    assert_eq!(parse_from!(ExprParser r#""\n""#), Ok("\n".into()));
}

#[test]
fn yakker_basic_single_quotes() {
    assert_eq!(parse_from!(RegularRightSideParser r"'c'"), Ok(RegularRightSide::Term("c".into())));
    assert_eq!(parse_from!(RegularRightSideParser r"'c''d'"),
               Ok(RegularRightSide::Concat(Box::new(RegularRightSide::Term("c".into())),
                                           Box::new(RegularRightSide::Term("d".into()))
               )));
    assert_eq!(parse_from!(RegularRightSideParser r"'('"), Ok(RegularRightSide::Term("(".into())));
    assert_eq!(parse_from!(RegularRightSideParser r"')'"), Ok(RegularRightSide::Term(")".into())));
}
#[test]
fn yakker_nonterm() {
    assert_eq!(parse_from!(NonTermParser r"A"), Ok("A".into()));
}

#[test]
fn yakker_rules() {
    assert_eq!(parse_from!(RuleParser r"A::= 'c'"), Ok(Rule::labelled_new("0:1", "A".into(), None, RegularRightSide::Term("c".into()))));
    assert_eq!(parse_from!(GrammarParser r"A::= 'c'"), Ok(Grammar { rules: vec![Rule::labelled_new("0:1", "A".into(), None, RegularRightSide::Term("c".into()))]}));
    assert_eq!(parse_from!(GrammarParser r"A::= 'a';B::= 'b'"), Ok(Grammar {
        rules: vec![
            Rule::labelled_new("0:1", "A".into(), None, RegularRightSide::Term("a".into())),
            Rule::labelled_new("9:10", "B".into(), None, RegularRightSide::Term("b".into()))]}));
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
    assert_matches!(parse_from!(NonTermParser "Int"), Ok(_));
    assert_matches!(parse_from!(RegularRightSideParser "<x:=Int(())>"), Ok(_));
    assert_matches!(parse_from!(RightSideLeafParser "'0'"), Ok(_));
    assert_matches!(parse_from!(RuleParser "Int ::= '0' "), Ok(_));
    assert_matches!(parse_from!(RuleParser "Int ::= ( ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') )* "), Ok(_));

    assert_matches!(parse_from!(RuleParser "Int ::= { n:=y_0 } ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0]"), Ok(_));
    assert_matches!(parse_from!(RuleParser "Int(n) ::= ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0]"), Ok(_));
}

#[cfg(test)]
fn imperative_fixed_width_integer_grammar() -> Grammar {
    let s = r"Int(n) ::= ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0];";
    let lex = toyman::Lexer::new(s);
    yakker::GrammarParser::new().parse(s, lex).unwrap()
}

#[test]
fn imperative_fixed_width_integer_1() {
    let g = imperative_fixed_width_integer_grammar();
    assert!(g.matches(&input("1"), &right_side(r"<Int(1)>")).has_parse());
    assert!(g.matches(&input("0"), &right_side(r"<Int(1)>")).has_parse());
}

#[test]
fn simpler_variant_on_ifwi2_a() {
    assert!(parse_from!(GrammarParser r"S(n) ::= [n gt 0] 'a' { n := n-1 } [n gt 0] 'b' { n := n-1 } [n eql 0];")
            .unwrap()
            .matches(&input("ab"), &right_side(r"<S(2)>"))
            .has_parse());
}

#[test]
fn simpler_variant_on_ifwi2_b() {
    assert!(parse_from!(GrammarParser r"S(n) ::= ([n gt 0] ( 'a' | 'b' ) { n := n- 1 })* [n eql 0];")
            .unwrap()
            .matches(&input("ab"), &right_side(r"<S(2)>"))
            .has_parse());
}

#[test]
fn imperative_fixed_width_integer_2() {
    let g = imperative_fixed_width_integer_grammar();
    assert!(g.matches(&input("10"), &right_side(r"<Int(2)>")).has_parse());
}

// Example: Functional fixed-width integer
//
//    dig = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
// int(n) = [n = 0] | [n > 0] dig int(n - 1)

#[test]
fn functional_fixed_width_integer() {
    let g = parse_from!(GrammarParser r"Dig ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'; Int(n) ::= [n eql 0] | [ n gt 0 ] Dig <Int(n - 1)>;").unwrap();
    assert!(g.matches(&input("0"), &right_side(r"<Int(1)>")).has_parse());
    assert!(g.matches(&input("1"), &right_side(r"<Int(1)>")).has_parse());
    assert!(g.matches(&input("10"), &right_side(r"<Int(2)>")).has_parse());
}

// Example: Left-factoring
//
// A = (B '?') | (C '!')
// B = 'x' '+' 'x'
// C = ('x' '+' 'x') | ('x' '-' 'x')

#[test]
fn left_factoring() {
    parse_from!(RuleParser "A ::= (B '?') | (C '!')").unwrap();
    let g = parse_from!(GrammarParser r"A ::= (B '?') | (C '!'); B ::= 'x' '+' 'x'; C ::= ('x' '+' 'x') | ('x' '-' 'x');").unwrap();
    assert!(g.matches(&input("x+x?"), &right_side("A")).has_parse());
    assert!(g.matches(&input("x+x!"), &right_side("A")).has_parse());
    assert!(g.matches(&input("x-x!"), &right_side("A")).has_parse());
    assert!(g.matches(&input("x-x?"), &right_side("A")).no_parse());
}

// Example: "Regular right sides"
//
//  digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
// number = digit digit*
#[test]
fn regular_right_sides_number_is_digit_digit_star() {
    let g = parse_from!(GrammarParser "Digit ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'; Number ::= Digit Digit*;").unwrap();
    assert!(g.matches(&input(""), &right_side("Digit")).no_parse());
    assert!(g.matches(&input(""), &right_side("Number")).no_parse());
    assert!(g.matches(&input("0"), &right_side("Digit")).has_parse());
    assert!(g.matches(&input("0"), &right_side("Number")).has_parse());
    assert!(g.matches(&input("01"), &right_side("Number")).has_parse());
    assert!(g.matches(&input("10"), &right_side("Number")).has_parse());
    assert!(g.matches(&input("98765433210"), &right_side("Number")).has_parse());
}

// Example: "Full context-free grammars"
//
// typename = identifer
//     expr = '&' expr               ; address of
//          | expr '&' expr          ; bitwise conjunction
//          | '(' typename ')' expr  ; cast
//          | '(' expr ')'
//          | identifier
//          | ...
//
// FIXME: at some point, I think I'll want to factor things so that a grammar
// can be independent of its lexer. I.e. that certain syntatic classes are
// treated as entirely handled by the lexing pass. (I had originally thought
// that I might handle this by having identifiers with an initial lower-case letter
// denote classes defined by the lexer, but now I remember that part of the design
// here, if I recall correctly, is that identifiers with an initial lower-case letter
// denote Yakker variables.
//
// In any case, for now we deal with this by just having a naive identifier
// grammar here in place.
#[test]
fn full_context_free_grammars() {
    let g = parse_from!(GrammarParser "Typename ::= Identifier; Expr ::= '&' Expr | Expr '&' Expr | '(' Typename ')' Expr | '(' Expr ')' | Identifier; Identifier ::= Letter Letter*; Letter ::= 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j' |'k' | 'l' | 'm' | 'n' | 'o' | 'p' | 'q' | 'r' | 's' | 't' | 'u' | 'v'| 'w' | 'x' | 'y' | 'z';").unwrap();
    assert!(g.matches(&input("x"), &right_side("Letter")).has_parse());
    assert!(g.matches(&input("x"), &right_side("Identifier")).has_parse());
    assert!(g.matches(&input("x"), &right_side("Expr")).has_parse());
}


// Example: "Attribute-directed parsing"
//
// As adapted from IMAP mail protocol 
//  literal8 = " ̃{" number ["+"] "}" CRLF *OCTET
//        ;; A string that might contain NULs.
//        ;; <number> represents the number of OCTETs
//        ;; in the response string.
//
// literal8 = '~' '{' x:=number ('+' | epsilon) '}' { n := string2int(x) } CRLF ([n > 0] OCTET { n := n - 1 })* [n = 0]
#[test]
fn attribute_directed_parsing() {
    let g = parse_from!(GrammarParser r#"Sdigit ::= '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'; Digit ::= '0' | Sdigit; Number ::= Sdigit Digit*; Octet ::= 'a' | 'b'; Literal8 ::= '~' '{' x := Number ( '+' | '' ) '}'  { n := string2int(x) } '\n' ([n > 0] Octet { n := n - 1 })* [n == 0]"#).unwrap();

    assert!(g.matches(&input("2"), &right_side("Number")).has_parse());
    assert!(g.matches(&input("2"), &right_side("x := Number")).has_parse());
    assert!(g.matches(&input("\n"), &right_side("'\n'")).has_parse());
    assert!(g.matches(&input("\n"), &right_side(r"'\n'")).has_parse());
    assert!(g.matches(&input("~{2}\nab"), &right_side("Literal8")).has_parse());
}


// Example: "Parmeterized Nonterminals"
//
// stringFW(n) = ([n > 0] CHAR8 { n := n-1 })* [n = 0]
#[test]
fn parameterized_non_terms_1() {
    let g = parse_from!(GrammarParser r#"StringFW(n) ::= ([n > 0] CHAR8 { n := n-1 })* [n == 0]; CHAR8 ::= 'a' | 'b' | 'c';"#).unwrap();
    // Note; we currently have to use `<NonTerm(arg_expr)>`, with explicit angle-brackets, for these
    // applications of parameterized non-terminals. The "obvious" way to fix the yakker.lalrpop
    // definition leads to a complaint that the resulting grammar is not LR(1) anymore, probably
    // because of the ambiguity in precedence with alternative construction, plus the fact that
    // non-terminals are allowed to occur on their own entirely.


    assert!(g.matches(&input("abc"), &right_side("<StringFW(3)>")).has_parse());
    assert!(g.matches(&input("abc"), &right_side("<StringFW(4)>")).no_parse());
    assert!(g.matches(&input("abca"), &right_side("<StringFW(4)>")).has_parse());
    assert!(g.matches(&input(""), &right_side("<StringFW(0)>")).has_parse());
}

// stringFW(n) = [n = 0] | [n > 0] CHAR8 stringFW(n - 1)
#[test]
fn parameterized_non_terms_2() {
    // Note; we currently have to use `<NonTerm(arg_expr)>`, with explicit angle-brackets, for these
    // applications of parameterized non-terminals. The "obvious" way to fix the yakker.lalrpop
    // definition leads to a complaint that the resulting grammar is not LR(1) anymore, probably
    // because of the ambiguity in precedence with alternative construction, plus the fact that
    // non-terminals are allowed to occur on their own entirely.

    let g = parse_from!(GrammarParser r#"StringFW(n) ::= [n == 0] | [n > 0] CHAR8 <StringFW(n - 1)>; CHAR8 ::= 'a' | 'b' | 'c';"#).unwrap();

    assert!(g.matches(&input("abc"), &right_side("<StringFW(3)>")).has_parse());
    assert!(g.matches(&input("abc"), &right_side("<StringFW(4)>")).no_parse());
    assert!(g.matches(&input("abca"), &right_side("<StringFW(4)>")).has_parse());
    assert!(g.matches(&input(""), &right_side("<StringFW(0)>")).has_parse());
    assert!(g.matches(&input("a"), &right_side("<StringFW(0)>")).has_parse());
}

#[test]
fn flexible_quote_delims() {
    let g = parse_from!(GrammarParser r##"TripleA ::= r#'abc'#;"##).unwrap();
    assert!(g.matches(&input("abc"), &right_side("TripleA")).has_parse());
}

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
