#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(super) struct MemOffset(usize);

impl MemOffset {
    pub const INVALID: Self = Self(usize::MAX);
    const BIT: usize = 1 << (usize::BITS - 1);

    #[inline]
    pub const fn from_static(offset: usize) -> Self {
        Self(offset)
    }

    #[inline]
    pub const fn from_stack(offset: usize) -> Self {
        Self(offset | Self::BIT)
    }

    #[inline]
    pub const fn is_invalid(self) -> bool {
        self.0 == Self::INVALID.0
    }

    #[inline]
    pub const fn is_static(self) -> bool {
        self.0 & Self::BIT == 0
    }

    #[inline]
    pub fn offset(self) -> usize {
        debug_assert_ne!(self.0, Self::INVALID.0);
        self.0 & !Self::BIT
    }
}
