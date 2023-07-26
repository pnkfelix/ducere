pub(crate) trait Bother<'a, T> { fn b_iter(self) -> Box<dyn Iterator<Item=T> + 'a>; }

impl<'a, T:'a> Bother<'a, T> for Option<T> {
    fn b_iter(self) -> Box<dyn Iterator<Item=T>+'a> {
        Box::new(self.into_iter())
    }
}
