//! bounded vocabulary indexes for experimental valid-token discovery.

use std::array;
use std::mem::size_of;

/// Token IDs grouped by the first byte of each non-empty vocabulary entry.
///
/// Every non-empty token ID appears exactly once. Buckets preserve ascending
/// token-ID order because the source vocabulary is indexed in ID order.
pub struct FirstByteTokenIndex {
    buckets: [Vec<usize>; 256],
    non_empty_token_count: usize,
    retained_bytes: usize,
}

impl FirstByteTokenIndex {
    pub fn new(vocabulary: &[Vec<u8>]) -> Self {
        let mut buckets: [Vec<usize>; 256] = array::from_fn(|_| Vec::new());
        let mut non_empty_token_count = 0;

        for (token_id, token_bytes) in vocabulary.iter().enumerate() {
            let Some(first_byte) = token_bytes.first() else {
                continue;
            };
            buckets[*first_byte as usize].push(token_id);
            non_empty_token_count += 1;
        }

        for bucket in &mut buckets {
            bucket.shrink_to_fit();
        }

        let retained_bytes = size_of::<Self>()
            + buckets
                .iter()
                .map(|bucket| bucket.capacity() * size_of::<usize>())
                .sum::<usize>();

        Self {
            buckets,
            non_empty_token_count,
            retained_bytes,
        }
    }

    pub fn candidate_count(&self, allowed_first_bytes: &[bool; 256]) -> usize {
        self.buckets
            .iter()
            .zip(allowed_first_bytes)
            .filter_map(|(bucket, allowed)| allowed.then_some(bucket.len()))
            .sum()
    }

    pub fn for_each_candidate(
        &self,
        allowed_first_bytes: &[bool; 256],
        mut visit: impl FnMut(usize),
    ) {
        for (bucket, allowed) in self.buckets.iter().zip(allowed_first_bytes) {
            if *allowed {
                for &token_id in bucket {
                    visit(token_id);
                }
            }
        }
    }

    pub fn non_empty_token_count(&self) -> usize {
        self.non_empty_token_count
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[cfg(test)]
    fn bucket(&self, first_byte: u8) -> &[usize] {
        &self.buckets[first_byte as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_each_non_empty_token_once_in_id_order() {
        let vocabulary = vec![
            Vec::new(),
            b"a".to_vec(),
            b"ab".to_vec(),
            b"b".to_vec(),
            vec![0xff],
            b"a".to_vec(),
        ];

        let index = FirstByteTokenIndex::new(&vocabulary);

        assert_eq!(index.non_empty_token_count(), 5);
        assert_eq!(index.bucket(b'a'), &[1, 2, 5]);
        assert_eq!(index.bucket(b'b'), &[3]);
        assert_eq!(index.bucket(0xff), &[4]);
        assert!(index.bucket(0).is_empty());
    }

    #[test]
    fn visits_only_allowed_buckets_and_reports_bounded_memory() {
        let vocabulary = vec![Vec::new(), b"a".to_vec(), b"b".to_vec(), b"a2".to_vec()];
        let index = FirstByteTokenIndex::new(&vocabulary);
        let mut allowed = [false; 256];
        allowed[b'a' as usize] = true;
        let mut candidates = Vec::new();

        index.for_each_candidate(&allowed, |token_id| candidates.push(token_id));

        assert_eq!(candidates, vec![1, 3]);
        assert_eq!(index.candidate_count(&allowed), 2);
        let minimum_bytes =
            std::mem::size_of_val(&index) + index.non_empty_token_count() * size_of::<usize>();
        assert!(index.retained_bytes() >= minimum_bytes);
        assert!(index.retained_bytes() < minimum_bytes + 1024);
    }
}
