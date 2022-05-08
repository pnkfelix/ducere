use crate::*;

fn lex(s: &str) -> impl Iterator<Item=Result<String, luthor::LexicalError>> + '_ {
    luthor::Lexer::new(s).map(|r|r.map(|(_, tok, _)|tok.into()))
}

fn lex_oks(s: &str) -> impl Iterator<Item=String> + '_ {
    luthor::Lexer::new(s).filter_map(|r| -> Option<String> { r.map(|(_, tok, _)|tok.into()).ok() })
}

fn strings<'a>(v: &'a [&str]) -> impl Iterator<Item=String> + 'a {
    v.iter().map(|s|s.to_string())
}

macro_rules! assert_strings_eq {
    ($lft:expr, $rgt:expr) => {
        // assert!($lft.eq($rgt))
        assert_eq!($lft.collect::<Vec<String>>(), $rgt.collect::<Vec<String>>())
    }
}

#[test]
fn lexing_basics() {
    assert!(lex("").eq(strings(&[]).map(|s|Ok(s))));
    assert_strings_eq!(lex_oks("a"), strings(&["a"]));
    assert_strings_eq!(lex_oks("ab"), strings(&["ab"]));
    assert_strings_eq!(lex_oks("a b"), strings(&["a", " ", "b"]));
    // this is more a meta-test: Its just pointing out that we'll be using raw-strings
    // in a lot of these tests, and that its important to not confuse the raw-strings
    // that *surround* the test input (which tend to have two or more #'s) with raw-strings
    // that are *in* the test input (which tend to have just zero or one #'s, though I did
    // use two #'s for cases where its matching non-quote forms internally.).
    assert_strings_eq!(lex_oks(r##"a b"##), strings(&["a", " ", "b"]));
    assert_strings_eq!(lex_oks(r##"a b c"##), strings(&["a", " ", "b", " ", "c"]));
}
#[test]
fn lexing_brackets() {
    assert_strings_eq!(lex_oks("(a b)"), strings(&["(", "a", " ", "b", ")"]));
    assert_strings_eq!(lex_oks("(a b"), strings(&["(", "a", " ", "b"]));
    assert_strings_eq!(lex_oks("((a b"), strings(&["(", "(", "a", " ", "b"]));
    assert_strings_eq!(lex_oks("([{a b"), strings(&["(", "[", "{", "a", " ", "b"]));
}
#[test]
fn lexing_quotations() {
    assert_strings_eq!(lex_oks(r##"a r#"b"# c"##), strings(&["a", " ", "b", " ", "c"]));
    assert_strings_eq!(lex_oks(r##"a r#"b b"# c"##), strings(&["a", " ", "b b", " ", "c"]));
    assert_strings_eq!(lex_oks(r##"a r#[b b]# c"##), strings(&["a", " ", "b b", " ", "c"]));

    assert_strings_eq!(lex_oks(r##"a r##[b [#] b]## c"##),
                       strings(&["a", " ", "b [#] b", " ", "c"]));

    assert_strings_eq!(lex_oks(r##"a r##[b #### [#] #### b]## c"##),
                       strings(&["a", " ", "b #### [#] #### b", " ", "c"]));

    assert_strings_eq!(lex_oks(r##"a r#[A]#r#(B)#r#{C}#r#"D"#r#'E'#r#<F>#r#|G|#r#`H`# c"##),
                       strings(&["a", " ", "A", "B", "C", "D", "E", "F", "G", "H", " ", "c"]));

    assert_strings_eq!(lex_oks(&[r##"a r[A(B)]r(C[D])r{[E](F)}r"{G}'H'""##,
                                 r##"r'I"J"'r#<KL>#r#|MN|#r#`OP`# c"##]
                               .into_iter().collect::<String>()),
                       strings(&["a", " ", "A(B)", "C[D]", "[E](F)", "{G}'H'",
                                 "I\"J\"", "KL", "MN", "OP", " ", "c"]));
}
