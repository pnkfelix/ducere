//! The Luthor lexer is very simple-minded. There are five categories
//! of "tokens":
//!
//! 1. whitespace: a string made up solely of whitespace characters
//!
//! 2. identifier word: a string made up solely of alphanumeric or underscore
//! characters, where the first character is alphabetic
//!
//! 3. numeric word: a string made up solely of alphanumeric or underscore
//! characters, where the first character is numeric
//!
//! 4. operative word: a string made up solely of non-whitespace,
//! non-alphanumeric characters
//!
//! 5. quotation: a string holding arbitrary content, delimited in a
//! content-dependent fashion. For the content-dependent delimiting, we extend
//! Rust's raw-string syntax: In addition to supporting quotes as the base
//! delimiters, we also support other "naturally" matching characters, yielding
//! e.g. "r#()#", "r[]", and "r##{}##". (As a special case, when content
//! permits, one can write the five cases of "", '', {}, [], and () without a
//! leading 'r' '#'^k, or with just a leading 'r' and no sharp signs '#'. But
//! all other extensions beyond, such as r##<>##, require the leading 'r' '#'^k
//! with k>=1.
//!
//! Aside: Many languages would break operators apart based on fine-grained
//! rules; and some languages like FORTH and Lisp/Scheme would allow identifiers
//! to have operator characters embedded within them. Luthor takes a middle
//! ground that is optimized for allowing a rich collection of operators

use std::iter::Peekable;
use std::str::CharIndices;

use crate::Spanned;
use derive_more::{AsRef};
use unicode_brackets::UnicodeBrackets;

#[derive(PartialEq, Eq, Debug, AsRef)]
pub struct Ident<S>(S);
#[derive(PartialEq, Eq, Debug, AsRef)]
pub struct Numeric<S>(S);
#[derive(PartialEq, Eq, Debug, AsRef)]
pub struct Operative<S>(S);

impl AsRef<str> for Ident<String> { fn as_ref(&self) -> &str { self.0.as_ref() } }

impl<IS: Into<String>> From<Operative<IS>> for String { fn from(x: Operative<IS>) -> String { x.0.into() } }
impl<IS: Into<String>> From<Numeric<IS>> for String { fn from(x: Numeric<IS>) -> String { x.0.into() } }
impl<IS: Into<String>> From<Ident<IS>> for String { fn from(x: Ident<IS>) -> String { x.0.into() } }

trait IsOperative { fn is_operative(self) -> bool; }
impl IsOperative for char {
    fn is_operative(self) -> bool {
        !self.is_alphanumeric() && !self.is_whitespace()
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum Word<S> {
    Op(Operative<S>),
    Num(Numeric<S>),
    Id(Ident<S>),
}

impl<IS> AsRef<str> for Word<IS> where IS: AsRef<str> {
    fn as_ref(&self) -> &str {
        match self {
            Word::Op(x) => x.as_ref().as_ref(),
            Word::Num(x) => x.as_ref().as_ref(),
            Word::Id(x) => x.as_ref().as_ref(),
        }
    }
}

impl<IS> From<Word<IS>> for String where IS: Into<String> {
    fn from(w: Word<IS>) -> String {
        match w {
            Word::Op(x) => x.into(),
            Word::Num(x) => x.into(),
            Word::Id(x) => x.into(),
        }
    }
}

#[derive(PartialEq, Eq, Debug, AsRef)]
pub struct Bracket<S>(S);

impl<IS> From<Bracket<IS>> for String where IS: Into<String> { fn from(x: Bracket<IS>) -> String { x.0.into() } }


#[derive(PartialEq, Eq, Debug, AsRef)]
pub struct Whitespace<S>(S);

impl<IS> From<Whitespace<IS>> for String where IS: Into<String> { fn from(x: Whitespace<IS>) -> String { x.0.into() } }

#[derive(PartialEq, Eq, Debug)]
pub struct Delims(char, char);

#[derive(PartialEq, Eq, Debug, AsRef)]
pub struct Quoted<S> {
    // If None, then htis is not a raw-string
    // If Some, then holds the number of sharps between the 'r' and the oepn delimiter.
    pub sharp_count: Option<usize>,
    pub delim: Delims,
    #[as_ref]
    pub content: S,
}

impl<IS: Into<String>> From<Quoted<IS>> for String {
    fn from(q: Quoted<IS>) -> String { q.content.into() }
}

fn simple_delimiter(c: char) -> Option<Vec<char>> {
    match c {
        '[' => Some(vec!['[', ']']),
        '(' => Some(vec!['(', ')']),
        '{' => Some(vec!['{', '}']),
        '\"' => Some(vec!['"']),
        '\'' => Some(vec!['\'']),
        _ => None,
    }
}

fn raw_quoted_opener(c: char) -> Option<Vec<char>> {
    let s = simple_delimiter(c);
    if s.is_some() { return s; }
    match c {
        // no matter what other extensions we add, we obviously cannot use # as
        // a quotation delimiter on its own, because it won't compose with r#.
        '#' => None,
        '<' => Some(vec!['<', '>']),
        '|' => Some(vec!['|']),
        '`' => Some(vec!['`']),
        // to be determined: should this accept arbitrary operatives? Or even
        // arbitrary non-sharp characters, e.g.: `r#rHellor#` (yikes).
        _ => None,
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum TokKind {
    Bracket,
    Word(Word<()>),
    Quote(Quoted<()>),
    Space,
}

#[derive(PartialEq, Eq, Debug)]
pub struct Tok<S>(TokKind, S);

impl Tok<(usize, usize)> {
    fn repackage<'a>(self, data: &'a str) -> (usize, Tok<&'a str>, usize) {
        let (i, j) = self.1;
        dbg!((i, Tok(self.0, &data[i..=j]), j))
    }
}

impl<IS> AsRef<str> for Tok<IS> where IS: AsRef<str>{
    fn as_ref(&self) -> &str {
        self.1.as_ref()
    }
}

impl<IS> From<Tok<IS>> for String where IS: Into<String> {
    fn from(tok: Tok<IS>) -> String { tok.1.into() }
}

#[derive(PartialEq, Eq, Debug)]
pub enum LexicalError {
    UnterminatedQuote,
    InvalidQuotationBaseDelimiter
}

pub struct Lexer<'input> {
    input: &'input str,
    chars: Peekable<CharIndices<'input>>,
}

impl<'input> Lexer<'input> {
    pub fn new(input: &'input str) -> Self {
        Lexer { input, chars: input.char_indices().peekable()}
    }
}

/// Very regular token matching (no extra context needed)
enum R {
    WordOp,
    WordNum,
    WordId,
    Bracket,
    Space,
}

/// when lexing 'r' '#'^i . '#'^j OPEN, sharp_count is i, and when we read the
/// remaining j and the immediately following open-delimiter, then the total
/// number of sharps we are seeking is k = i+j.
struct QuoteOpen {
    sharp_count: usize,
}

/// after lexing 'r' '#'^k OPEN, then we are seeking CLOSE '#'^k to end the
/// raw-quoted content.
#[derive(Debug)]
struct QuoteClose {
    sharp_seek: usize,
    sharp_count: usize,
    end_delim_set: Vec<char>
}

// Matching overall; cases subdivided by control complexity
// enum M { Reg(R), Quo(Q), }
enum RegAction { Complete, Continue, }
impl R {
    fn from_start_char(c: char) -> Self {
        if c.is_numeric() { R::WordNum }
        else if c.is_alphabetic() { R::WordId }
        else if c.is_whitespace() { R::Space }
        else if c.is_open_bracket() || c.is_close_bracket() { R::Bracket }
        else { R::WordOp }
    }
    fn action(&self, p: char) -> RegAction {
        match self {
            // every bracket is its own token; we don't merge sequences of brackets into one token.
            R::Bracket => { RegAction::Complete }
            R::WordNum => if p.is_alphanumeric() { RegAction::Continue } else { RegAction::Complete }
            R::WordId => if p.is_alphanumeric() { RegAction::Continue } else { RegAction::Complete }
            R::WordOp => if p.is_operative() { RegAction::Continue } else { RegAction::Complete }
            R::Space => if p.is_whitespace() { RegAction::Continue } else { RegAction::Complete }
        }
    }
}

impl R {
    fn finalize(&self, span: (usize, usize)) -> Result<Tok<(usize, usize)>, LexicalError> {
        match self {
            R::Bracket => Ok(Tok(TokKind::Bracket, span)),
            R::WordOp => Ok(Tok(TokKind::Word(Word::Op(Operative(()))), span)),
            R::WordNum => Ok(Tok(TokKind::Word(Word::Num(Numeric(()))), span)),
            R::WordId => Ok(Tok(TokKind::Word(Word::Id(Ident(()))), span)),
            R::Space => Ok(Tok(TokKind::Space, span)),
        }
    }
}

impl<'input> Lexer<'input> {
    fn read_regular(&mut self, mut ic: (usize, char)) -> Option<<Self as Iterator>::Item> {
        let (i, c) = ic;
        let spanned_start = i;
        let r = R::from_start_char(c);
        let data = self.input;
        let mut buf = String::new();
        buf.push(c);
        loop {
            let (i, _) = ic;
            let p: Option<char> = self.chars.peek().map(|(_, c)|*c);
            let p = match p {
                // we're done reading input; finalize and return this token.
                None => return Some(r.finalize((spanned_start, i)).map(|t|t.repackage(data))),
                Some(p) => p,
            };
            match r.action(p) {
                RegAction::Continue => {
                    ic = self.chars.next().unwrap();
                    dbg!(ic);
                    let c = ic.1;
                    assert_eq!(c, p);
                    buf.push(c);
                    continue;
                }
                RegAction::Complete => {
                    return Some(r.finalize((spanned_start, i)).map(|t|t.repackage(data)));
                }
            }
        }
    }

    fn read_raw_quotation(&mut self, ic: (usize, char), mut p: char) -> Option<<Self as Iterator>::Item> {
        let (i, c) = ic;
        let spanned_start = i;
        assert_eq!(c, 'r');
        let mut sharp_count = 0;

        // First: count how many sharps we are matching.
        while p == '#' {
            // assert_eq!(p, '#');
            let ic = self.chars.next().unwrap();
            let c = ic.1;
            assert_eq!(c, p);
            sharp_count += 1;
            let opt_p = self.chars.peek().map(|(_, c)|*c);
            match opt_p {
                None => break,
                Some(p_) => {
                    p = p_;
                }
            }
        }
        let q = QuoteOpen { sharp_count };

        // now: c is either 'r' or '#', and p is the base quotation delimiter.
        let base_open_delim;
        let content_start;
        let end_delim_set = if let Some(end) = raw_quoted_opener(p) {
            let ic = self.chars.next().unwrap();
            content_start = ic.0+1;
            let c = ic.1;
            assert_eq!(c, p);
            base_open_delim = c;
            end
        } else {
            return Some(Err(LexicalError::InvalidQuotationBaseDelimiter));
        };
        let mut q = QuoteClose {
            sharp_seek: q.sharp_count,
            sharp_count: 0,
            end_delim_set,
        };

        let mut buf = String::new();

        loop {
            let opt_ic = self.chars.next();
            let ic = match opt_ic {
                None => return Some(Err(LexicalError::UnterminatedQuote)),
                Some(ic) => ic,
            };
            let c = ic.1;

            if q.end_delim_set.contains(&c) {
                // this *might* be the end of the quotation. If we can read
                // required number of sharp characters, then return the
                // quotation token. iF we cannot read the required number of
                // tokens, then add the saved potential end-delimiting character
                // `c` as well as the number of sharps that *were* read, and
                // then continue on accumulating into the quotation.
                let content_end = ic.0-1;
                loop {
                    if q.sharp_seek == q.sharp_count {
                        // we got to the number of sharps sought,
                        // then we have a full quotation form.
                        let tok = Tok(TokKind::Quote(Quoted {
                            sharp_count: Some(q.sharp_seek),
                            delim: Delims(base_open_delim, c),
                            content: (),
                        }), &self.input[content_start..=content_end]);
                        return Some(Ok((spanned_start, tok, i)));
                    }
                    let opt_ip = self.chars.peek().cloned();
                    if opt_ip.map(|(_, p)|p) == Some('#') {
                        let opt_ic = self.chars.next();
                        assert_eq!(opt_ip, opt_ic);
                        q.sharp_count += 1;
                        continue;
                    }

                    // if we get here, then we have insufficent sharps
                    // and the peeked character is not a sharp; therefore
                    // `c` is not the actual end-delimiter for this quotation,
                    // adn we should continue scanning. But before we
                    // do that, we need to copy everyything we scanned
                    // into the buffer.

                    buf.push(c);
                    for _ in 0..q.sharp_count { buf.push('#'); }
                    q.sharp_count = 0;
                    break;
                }
            } else {
                // if `c` is not in the end delimiter set, then we can just
                // accumulate it and move on.
                buf.push(c);
                continue;
            }
        }
    }
}

impl<'input> Iterator for Lexer<'input> {
    type Item = Spanned<Tok<&'input str>, usize, LexicalError>;
    fn next(&mut self) -> Option<Self::Item> {
        let (i, c): (usize, char) = match self.chars.next() {
            Some((i, c)) => (i,c),
            None => return None, // End of file
        };
        let p: Option<char> = self.chars.peek().map(|(_, c)|*c);
        match (c, p) {
            ('r', Some(p)) if (p == '#' || raw_quoted_opener(p).is_some()) => {
                self.read_raw_quotation((i, c), p)
            }
            _ => self.read_regular((i, c))
        }
    }
}
