use super::{luthor, Spanned, YakkerError};

pub struct Lexer<'a>(luthor::Lexer<'a>);

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer(luthor::Lexer::new(input))
    }
}

impl<'input> Iterator for Lexer<'input> {
    type Item = Spanned<Tok<'input>, usize, YakkerError>;

    fn next(&mut self) -> Option<Self::Item> {
        use luthor::{Delims, Word, Quoted};
        use luthor::TokKind as K;

        loop {
            let x = if let Some(x) = nbg!(self.0.next()) { x } else { return None; };
            let (i, x, j) = match x { Ok(x) => nbg!(x), Err(e) => { return Some(Err(YakkerError::Lex(e))); } };
            let opt_c = nbg!(x.data()).chars().next();
            let tok = match (*x.data(), x.kind()) {
                (s, K::Bracket) => {
                    Tok::Bracket(s)
                }

                (s, K::Word(Word::Com(_))) => {
                    Tok::Commalike(s)
                }

                (s, K::Word(Word::Num(_))) => {
                    Tok::Numeric(s)
                }

                (s, K::Word(Word::Op(_))) => {
                    match s {
                        _ => Tok::Operative(s)
                    }
                }

                (s, K::Quote(Quoted { sharp_count: _, delim: Delims(c1, c2), content: _ })) => {
                    nbg!(Tok::QuoteLit(*c1, s, *c2))
                }

                (_, K::Space) => {
                    // skip the space and grab next token.
                    continue;
                }

                (s, K::Word(Word::Id(_))) if s.chars().next().unwrap().is_uppercase() => {
                    nbg!(Tok::UpperIdent(s))
                }
                (s, K::Word(Word::Id(_))) => {
                    let c = opt_c.unwrap();
                    assert!(c == '_' || c.is_lowercase());
                    Tok::LowerIdent(s)
                }
            };
            return Some(Ok(nbg!((i, tok, j))));
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tok<'a> {
    // "(", ")", "{", "}", "[", "]"
    Bracket(&'a str),

    // r"[a-z_][a-zA-Z_0-9]*"
    LowerIdent(&'a str),
    // r"[A-Z][a-zA-Z_0-9]*"
    UpperIdent(&'a str),
    // r"[1-9][0-9]*|0"
    Numeric(&'a str),

    // ";" or ","
    Commalike(&'a str),

    // r#""(\\"|[^"])*""# => STRING_LIT
    // r"'[^'\\]'"
    QuoteLit(char, &'a str, char),

    // (other)
    Operative(&'a str),
}
