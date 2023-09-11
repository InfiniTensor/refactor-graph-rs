use std::{alloc::Layout, ptr::NonNull};

#[derive(PartialEq, Eq, Debug)]
pub struct Blob {
    ptr: Option<NonNull<u8>>,
    layout: Layout,
}

impl Blob {
    pub const EMPTY: Self = Self {
        ptr: None,
        layout: Layout::new::<u8>(),
    };

    /// Creates a new blob with data allocated.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the data is valid.
    #[inline]
    pub unsafe fn new(layout: Layout, data: *const u8) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(std::alloc::alloc(layout)) };
        unsafe { ptr.as_ptr().copy_from_nonoverlapping(data, layout.size()) };
        Self {
            ptr: Some(ptr),
            layout,
        }
    }

    /// Checks if the blob has data allocated.
    #[inline]
    pub const fn exist(&self) -> bool {
        self.ptr.is_some()
    }

    /// Gets the pointer to the data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the blob has data allocated.
    #[inline]
    pub unsafe fn as_ptr_unchecked(&self) -> *const u8 {
        self.ptr.unwrap().as_ptr()
    }

    /// Gets the mutable pointer to the data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the blob has data allocated.
    #[inline]
    pub unsafe fn as_mut_ptr_unchecked(&self) -> *mut u8 {
        self.ptr.unwrap().as_ptr()
    }
}

impl Drop for Blob {
    #[inline]
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            unsafe { std::alloc::dealloc(ptr.as_ptr(), self.layout) };
        }
    }
}
