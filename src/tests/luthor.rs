use crate::*;

fn lex(s: &str) -> impl Iterator<Item=Result<String, LexicalError>> + '_ {
    Lexer::new(s).map(|r|r.map(|(_, tok, _)|tok.into()))
}

fn lex_oks(s: &str) -> impl Iterator<Item=String> + '_ {
    Lexer::new(s).filter_map(|r| -> Option<String> { r.map(|(_, tok, _)|tok.into()).ok() })
}

fn strings<'a>(v: &'a [&str]) -> impl Iterator<Item=String> + 'a {
    v.iter().map(|s|s.to_string())
}

#[test]
fn lexing_basics() {
    assert!(lex("").eq(strings(&[]).map(|s|Ok(s))));
    assert!(lex_oks("a").eq(strings(&["a"])));
    assert!(lex_oks("ab").eq(strings(&["ab"])));
    assert!(lex_oks("a b").eq(strings(&["a", " ", "b"])));
    assert!(lex_oks(r##"a b"##).eq(strings(&["a", " ", "b"])));
    assert!(lex_oks(r##"a b c"##).eq(strings(&["a", " ", "b", " ", "c"])));
    assert!(lex_oks(r##"a r#"b"# c"##).eq(strings(&["a", " ", "b", " ", "c"])));
    assert!(lex_oks(r##"a r#"b b"# c"##).eq(strings(&["a", " ", "b b", " ", "c"])));
    assert!(lex_oks(r##"a r#[b b]# c"##).eq(strings(&["a", " ", "b b", " ", "c"])));

    assert!(lex_oks(r##"a r##[b [#] b]## c"##)
            .eq(strings(&["a", " ", "b [#] b", " ", "c"])));

    assert!(lex_oks(r##"a r##[b #### [#] #### b]## c"##)
            .eq(strings(&["a", " ", "b #### [#] #### b", " ", "c"])));

    assert!(lex_oks(r##"a r#[A]#r#(B)#r#{C}#r#"D"#r#'E'#r#<F>#r#|G|#r#`H`# c"##)
            .eq(strings(&["a", " ", "A", "B", "C", "D", "E", "F", "G", "H", " ", "c"])));

    assert!(lex_oks(r##"a r[AB]r(CD)r{EF}r"GH"r'IJ'r#<KL>#r#|MN|#r#`OP`# c"##)
            .eq(strings(&["a", " ", "AB", "CD", "EF", "GH", "IJ", "KL", "MN", "OP", " ", "c"])));
}
