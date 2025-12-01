use std::hint;

use arrow::array::{BooleanArray, FixedSizeListArray, ListArray, ValueSize};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field};
use arrow::offset::{Offsets, OffsetsBuffer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hint::black_box;
use polars_core::datatypes::{ArrayChunked, DataType, ListChunked};
use polars_core::frame::column::{Column, IntoColumn};
use polars_core::series::{IntoSeries, Series};
use polars_ops::chunked_array::ListNameSpaceImpl;
use polars_ops::series::concat_arr::concat_arr;
use polars_utils::pl_str::PlSmallStr;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn create_list_chunked_for_test(
    values_len: u64,
    item_capacity: u64,
    columns_count: u64,
) -> impl Iterator<Item = ListChunked> {
    let offsets: Vec<_> = (0..values_len as i64 + 1)
        .step_by(item_capacity as usize)
        .collect();

    (0..columns_count).map(move |index| {
        let mut range = SmallRng::seed_from_u64(index);

        let bool_iterator = range
            .clone()
            .random_iter::<bool>()
            .take(values_len as usize);

        let array = BooleanArray::new(
            ArrowDataType::Boolean,
            Bitmap::from_iter(bool_iterator),
            None,
        );

        let list_array = ListArray::new(
            ArrowDataType::LargeList(Box::new(Field::new(
                PlSmallStr::EMPTY,
                ArrowDataType::Boolean,
                true,
            ))),
            OffsetsBuffer::try_from(offsets.clone()).unwrap(),
            Box::new(array),
            None,
        );

        ListChunked::with_chunk(PlSmallStr::EMPTY, list_array)
    })
}

fn create_array_chunked_for_test(
    values_len: u64,
    item_capacity: u64,
    columns_count: u64,
) -> impl Iterator<Item = ArrayChunked> {
    (0..columns_count).map(move |index| {
        let mut range = SmallRng::seed_from_u64(index);

        let bool_iterator = range
            .clone()
            .random_iter::<bool>()
            .take(values_len as usize);

        let array = BooleanArray::new(
            ArrowDataType::Boolean,
            Bitmap::from_iter(bool_iterator),
            None,
        );

        let fs_list_array = FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(
                Box::new(Field::new(PlSmallStr::EMPTY, ArrowDataType::Boolean, false)),
                item_capacity as usize,
            ),
            array.len() / item_capacity as usize,
            Box::new(array),
            None,
        );

        ArrayChunked::with_chunk(PlSmallStr::EMPTY, fs_list_array)
    })
}
fn bench_concat_list_and_array(c: &mut Criterion) {
    let columns_count = 10;
    let values_len = 10_000;
    let mut group = c.benchmark_group("concat_lst");
    for item_capacity in [1, 10, 100, 1000, 10000].iter() {
        let mut ca_lists = create_list_chunked_for_test(values_len, *item_capacity, columns_count);
        let first_list = ca_lists.next().unwrap();
        let other_series: Vec<_> = ca_lists.map(|ca| ca.into_column()).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("list-{item_capacity}")),
            &(first_list, other_series.as_slice()),
            |b, (f, o)| b.iter(|| f.lst_concat(o)),
        );

        let ca_arrays: Vec<_> =
            create_array_chunked_for_test(values_len, *item_capacity, columns_count)
                .map(|item| item.into_column())
                .collect::<Vec<_>>();
        println!(
            "ca arrays: {:?}",
            ca_arrays
                .iter()
                .map(|v| v.array().unwrap().len())
                .collect::<Vec<_>>()
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("array-{item_capacity}")),
            ca_arrays.as_slice(),
            |b, arrays| {
                b.iter(|| {
                    concat_arr(
                        arrays,
                        &DataType::Array(
                            Box::new(DataType::Boolean),
                            (*item_capacity * columns_count) as usize,
                        ),
                    )
                })
            },
        );
    }
}

criterion_group!(benches, bench_concat_list_and_array);
criterion_main!(benches);
