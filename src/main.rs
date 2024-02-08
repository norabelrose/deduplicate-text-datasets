/* Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Create and use suffix arrays for deduplicating language model datasets.
 *
 * A suffix array A for a sequence S is a datastructure that contains all
 * suffixes of S in sorted order. To be space efficient, instead of storing
 * the actual suffix, we just store the pointer to the start of the suffix.
 * To be time efficient, it uses fancy algorithms to not require quadratic
 * (or worse) work. If we didn't care about either, then we could literally
 * just define (in python)
 * A = sorted(S[i:] for i in range(len(S)))
 *
 * Suffix arrays are amazing because they allow us to run lots of string
 * queries really quickly, while also only requiring an extra 8N bytes of
 * storage (one 64-bit pointer for each byte in the sequence).
 *
 * This code is designed to work with Big Data (TM) and most of the
 * complexity revolves around the fact that we do not require the
 * entire suffix array to fit in memory. In order to keep things managable,
 * we *do* require that the original string fits in memory. However, even
 * the largest language model datasets (e.g., C4) are a few hundred GB
 * which on todays machines does fit in memory.
 *
 * With all that amazing stuff out of the way, just a word of warning: this
 * is the first program I've ever written in rust. I still don't actually
 * understand what borrowing something means, but have found that if I
 * add enough &(&&x.copy()).clone() then usually the compiler just loses
 * all hope in humanity and lets me do what I want. I apologize in advance
 * to anyone actually does know rust and wants to lock me in a small room
 * with the Rust Book by Klabnik & Nichols until I repent for my sins.
 * (It's me, two months in the future. I now more or less understand how
 * to borrow. So now instead of the code just being all awful, you'll get
 * a nice mix of sane rust and then suddenly OH-NO-WHAT-HAVE-YOU-DONE-WHY!?!)
 */

use std::cmp::Reverse;
use std::convert::TryInto;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Read;
// use std::path;
use std::path::Path;
use std::time::Instant;
use std::thread;

extern crate clap;
extern crate filebuffer;
extern crate rayon;

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

mod table;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(arg_required_else_help = true)]
    Make {
        #[clap(short, long)]
        data_file: String,
    },

    MakePart {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        start_byte: usize,
        #[clap(short, long)]
        end_byte: usize,
    },

    CountOccurrences {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        query_file: String,
        #[clap(short, long)]
        print_location: bool,
        #[clap(short, long)]
        load_disk: bool,
    },

    CountOccurrencesMulti {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        query_file: String,
        #[clap(short, long)]
        load_disk: bool,
    },

    Merge {
        #[clap(short, long)]
        suffix_path: Vec<String>,
        #[clap(short, long)]
        output_file: String,
        #[clap(short, long, default_value_t = 8)]
        num_threads: i64,
    },

    Collect {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        cache_dir: String,
        #[clap(short, long)]
        length_threshold: u64,
    },
}

/// Return a zero-copy view of the given slice with the given type.
/// The resulting view has the same lifetime as the provided slice.
#[inline]
pub fn transmute_slice<'a, T, U>(slice: &'a [T]) -> &'a [U] {
    // SAFETY: We use floor division to ensure that we can't read past the end of the slice.
    let new_len = (slice.len() * std::mem::size_of::<T>()) / std::mem::size_of::<U>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const U, new_len) }
}

pub fn transmute_vec<'a, T, U>(vec: Vec<T>) -> &'a [U] {
    // SAFETY: We use floor division to ensure that we can't read past the end of the slice.
    let new_len = (vec.len() * std::mem::size_of::<T>()) / std::mem::size_of::<U>();
    let inner = vec.into_boxed_slice();

    unsafe { std::slice::from_raw_parts(Box::into_raw(inner) as *const U, new_len) }
}

/* Convert a uint64 array to a uint8 array.
 * This doubles the memory requirements of the program, but in practice
 * we only call this on datastructures that are smaller than our assumed
 * machine memory so it works.
 */
pub fn to_bytes(input: &[u64], size_width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_width * input.len());

    for value in input {
        bytes.extend(&value.to_le_bytes()[..size_width]);
    }
    bytes
}

/* Convert a uint8 array to a uint64. Only called on (relatively) small files. */
pub fn from_bytes(input: Vec<u8>, size_width: usize) -> Vec<u64> {
    println!("S {}", input.len());
    assert!(input.len() % size_width == 0);
    let mut bytes: Vec<u64> = Vec::with_capacity(input.len() / size_width);

    let mut tmp = [0u8; 8];
    // todo learn rust macros, hope they're half as good as lisp marcos
    // and if they are then come back and optimize this
    for i in 0..input.len() / size_width {
        tmp[..size_width].copy_from_slice(&input[i * size_width..i * size_width + size_width]);
        bytes.push(u64::from_le_bytes(tmp));
    }

    bytes
}

/* For a suffix array, just compute A[i], but load off disk because A is biiiiiiigggggg. */
fn table_load_disk(table: &mut BufReader<File>, index: usize, size_width: usize) -> usize {
    table
        .seek(std::io::SeekFrom::Start((index * size_width) as u64))
        .expect("Seek failed!");
    let mut tmp = [0u8; 8];
    table.read_exact(&mut tmp[..size_width]).unwrap();
    return u64::from_le_bytes(tmp) as usize;
}

/* Binary search to find where query happens to exist in text */
fn off_disk_position(
    text: &[u16],
    table: &mut BufReader<File>,
    query: &[u16],
    size_width: usize,
) -> usize {
    let (mut left, mut right) = (0, text.len());
    while left < right {
        let mid = (left + right) / 2;
        if query < &text[table_load_disk(table, mid, size_width)..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

/*
 * We're going to work with suffix arrays that are on disk, and we often want
 * to stream them top-to-bottom. This is a datastructure that helps us do that:
 * we read 1MB chunks of data at a time into the cache, and then fetch new data
 * when we reach the end.
 */
struct TableStream {
    file: BufReader<File>,
    cache: [u8; 8],
    size_width: usize,
}

/* Make a table from a file path and a given offset into the table */
fn make_table(path: std::string::String, offset: usize, size_width: usize) -> TableStream {
    let mut table = TableStream {
        file: std::io::BufReader::with_capacity(1024 * 1024, fs::File::open(path).unwrap()),
        cache: [0u8; 8],
        size_width: size_width,
    };
    table
        .file
        .seek(std::io::SeekFrom::Start((offset * size_width) as u64))
        .expect("Seek failed!");
    return table;
}

/* Get the next word from the suffix table. */
fn get_next_pointer_from_table_canfail(tablestream: &mut TableStream) -> u64 {
    let ok = tablestream
        .file
        .read_exact(&mut tablestream.cache[..tablestream.size_width]);
    let bad = match ok {
        Ok(_) => false,
        Err(_) => true,
    };
    if bad {
        return std::u64::MAX;
    }
    let out = u64::from_le_bytes(tablestream.cache);
    return out;
}

fn table_load_filebuffer(table: &filebuffer::FileBuffer, index: usize, width: usize) -> usize {
    let mut tmp = [0u8; 8];
    tmp[..width].copy_from_slice(&table[index * width..index * width + width]);
    return u64::from_le_bytes(tmp) as usize;
}

fn table_load(table: &[u8], index: usize, width: usize) -> usize {
    let mut tmp = [0u8; 8];
    tmp[..width].copy_from_slice(&table[index * width..index * width + width]);
    return u64::from_le_bytes(tmp) as usize;
}

/*
 * Helper function to actually do the count of the number of times something is repeated.
 * This should be fairly simple.
 * First, perform binary search using the on-disk suffix array to find the first place
 * where the string occurrs. If it doesn't exist then return 0.
 * Then, binary search again to find the last location it occurrs.
 * Return the difference between the two.
 */
fn count_occurances(
    text: &filebuffer::FileBuffer,
    size_text: u64,
    table: &filebuffer::FileBuffer,
    size: u64,
    str: &[u8],
    size_width: usize,
    print_where: bool,
) -> u64 {
    let mut buf: &[u8];
    assert!(size % (size_width as u64) == 0);

    let mut low = 0;
    let mut high = size / (size_width as u64);
    while low < high {
        let mid = (high + low) / 2;
        let pos = table_load_filebuffer(&table, mid as usize, size_width);

        if pos + str.len() < size_text as usize {
            buf = &text[pos..pos + str.len()];
        } else {
            buf = &text[pos..size_text as usize];
        }

        if str <= &buf {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let start = low;

    let pos = table_load_filebuffer(&table, low as usize, size_width);
    if pos + str.len() < size_text as usize {
        buf = &text[pos..pos + str.len()];
    } else {
        buf = &text[pos..size_text as usize];
    }

    if str != buf {
        return 0; // not found
    }

    high = size / (size_width as u64);
    while low < high {
        let mid = (high + low) / 2;
        let pos = table_load_filebuffer(&table, mid as usize, size_width);

        if pos + str.len() < size_text as usize {
            buf = &text[pos..pos + str.len()];
        } else {
            buf = &text[pos..size_text as usize];
        }

        if str != buf {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    if print_where {
        for i in start..low {
            let pos = table_load_filebuffer(&table, i as usize, size_width);
            println!("Found at: {}", pos);
            break;
        }
    }

    return low - start;
}

fn count_occurances_memory(
    text: &[u8],
    size_text: u64,
    table: &[u8],
    size: u64,
    str: &[u8],
    size_width: usize,
    print_where: bool,
) -> u64 {
    let mut buf: &[u8];
    assert!(size % (size_width as u64) == 0);

    let mut low = 0;
    let mut high = size / (size_width as u64);
    while low < high {
        let mid = (high + low) / 2;
        let pos = table_load(&table, mid as usize, size_width);

        if pos + str.len() < size_text as usize {
            buf = &text[pos..pos + str.len()];
        } else {
            buf = &text[pos..size_text as usize];
        }

        if str <= &buf {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let start = low;

    let pos = table_load(&table, low as usize, size_width);
    if pos + str.len() < size_text as usize {
        buf = &text[pos..pos + str.len()];
    } else {
        buf = &text[pos..size_text as usize];
    }

    if str != buf {
        return 0; // not found
    }

    high = size / (size_width as u64);
    while low < high {
        let mid = (high + low) / 2;
        let pos = table_load(&table, mid as usize, size_width);

        if pos + str.len() < size_text as usize {
            buf = &text[pos..pos + str.len()];
        } else {
            buf = &text[pos..size_text as usize];
        }

        if str != buf {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    if print_where {
        for i in start..low {
            let pos = table_load(&table, i as usize, size_width);
            println!("Found at: {}", pos);
            break;
        }
    }

    return low - start;
}

/*
 * Create a suffix array for a given file in one go.
 * Calling this method is memory heavy---it's technically linear in the
 * length of the file, but the constant is quite big.
 * As a result, this method should only be called for files that comfortably
 * fit into memory.
 *
 * The result of calling this method is a new file with ".table.bin" appended
 * to the name which is the suffix array of sorted suffix pointers. This file
 * should be at most 8x larger than the original file (one u64 pointer per
 * byte of the original). In order to save space, if it turns out we only need
 * 32 bits to uniquely point into the data file then we serialize using fewer
 * bits (or 24, or 40, or ...), but in memory we always use a u64.
 *
 * If the file does not fit into memory, then instead you should use the
 * alternate save_part and then merge_parallel in two steps. See the comments
 * below for how those work.
 */
fn cmd_make(fpath: &String) -> std::io::Result<()> {
    let now = Instant::now();
    println!(
        "Reading the dataset at time t={}ms",
        now.elapsed().as_millis()
    );

    let text = fs::read(fpath.clone())?;
    println!(
        "Done reading the dataset at time t={}ms",
        now.elapsed().as_millis()
    );

    println!("... and now starting the suffix array construction.");

    let st = table::SuffixTable::new(transmute_slice(text.as_slice()));
    println!(
        "Done building suffix array at t={}ms",
        now.elapsed().as_millis()
    );
    let (_, table) = st.into_parts();

    let ratio = ((text.len() as f64).log2() / 8.0).ceil() as usize;
    println!("Ratio: {}", ratio);

    let mut buffer = File::create(fpath.clone() + ".table.bin")?;
    let bufout = to_bytes(&table, ratio);
    println!(
        "Writing the suffix array at time t={}ms",
        now.elapsed().as_millis()
    );
    buffer.write_all(&bufout)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}

/*
 * Create a suffix array for a subsequence of bytes.
 * As with save, this method is linear in the number of bytes that are
 * being saved but the constant is rather high. This method does exactly
 * the same thing as save except on a range of bytes.
 */
fn cmd_make_part(fpath: &String, start: u64, end: u64) -> std::io::Result<()> {
    let now = Instant::now();
    println!("Opening up the dataset files");

    let space_available = std::fs::metadata(fpath.clone())?.len() as u64;
    assert!(start < end);
    assert!(end <= space_available);

    let mut text = vec![0u8; (end - start) as usize];
    let mut file = fs::File::open(fpath.clone())?;
    println!("Loading part of file from byte {} to {}", start, end);
    file.seek(std::io::SeekFrom::Start(start))?;
    file.read_exact(&mut text)?;

    println!(
        "Done reading the dataset at time t={}ms",
        now.elapsed().as_millis()
    );
    println!("... and now starting the suffix array construction.");

    let st = table::SuffixTable::new(transmute_slice(text.as_slice()));
    println!(
        "Done building suffix array at t={}ms",
        now.elapsed().as_millis()
    );
    let (_, table) = st.into_parts();

    let ratio = ((text.len() as f64).log2() / 8.0).ceil() as usize;
    println!("Ratio: {}", ratio);

    let mut buffer = File::create(format!("{}.part.{}-{}.table.bin", fpath, start, end))?;
    let mut buffer2 = File::create(format!("{}.part.{}-{}", fpath, start, end))?;
    let bufout = to_bytes(&table, ratio);
    println!(
        "Writing the suffix array at time t={}ms",
        now.elapsed().as_millis()
    );
    buffer.write_all(&bufout)?;
    buffer2.write_all(&text)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}

/*
 * Count how many times a particular string has occurred in the dataset.
 *
 * This is the easiest method to understand. It just performs binary search on the
 * suffix array and uses it exactly as it was designed. It will output the number of counts.
 *
 * NOTE: This function allows overlapping sequences to count as different duplicates.
 * So if our string is `aaaa` and we count how many times `aa` occurrs, it will return 3,
 * not 2. This is different from python's "aaaa".count("aa") which will say 2.
 * This may or may not be a problem for you. But if is is, that's your problem, not mine.
 */
fn cmd_count_occurrences(
    fpath: &String,
    querypath: &String,
    print_location: bool,
    load_disk: bool,
) -> std::io::Result<()> {
    /* Count the number of times a particular sequence occurs in the table. */

    let metadata_text = fs::metadata(format!("{}", fpath))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", fpath))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    let mut str = Vec::with_capacity(std::fs::metadata(querypath.clone())?.len() as usize);
    fs::File::open(querypath.clone())?.read_to_end(&mut str)?;

    let occurances;

    if load_disk {
        let text = filebuffer::FileBuffer::open(fpath)?;
        let table = filebuffer::FileBuffer::open(format!("{}.table.bin", fpath))?;

        assert!(size_table % size_text == 0);
        let size_width = size_table / size_text;

        occurances = count_occurances_memory(
            &text,
            size_text,
            &table,
            size_table,
            &str[0..str.len()],
            size_width as usize,
            print_location,
        );
    } else {
        let mut text = Vec::with_capacity(size_text as usize);
        fs::File::open(format!("{}", fpath))?.read_to_end(&mut text)?;

        let mut table = Vec::with_capacity(size_table as usize);
        fs::File::open(format!("{}.table.bin", fpath))?.read_to_end(&mut table)?;

        assert!(size_table % size_text == 0);
        let size_width = size_table / size_text;

        occurances = count_occurances_memory(
            &text,
            size_text,
            &table,
            size_table,
            &str[0..str.len()],
            size_width as usize,
            print_location,
        );
    }

    println!("Number of times present: {}\n", occurances);
    Ok(())
}

/*
 * Count the number of times a particular sequence occurs in the table.
 * (for multiple queries)
 */
fn cmd_count_occurrences_multi(
    fpath: &String,
    querypath: &String,
    load_disk: bool,
) -> std::io::Result<()> {
    let metadata_text = fs::metadata(format!("{}", fpath))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", fpath))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    let mut str = Vec::with_capacity(std::fs::metadata(querypath.clone())?.len() as usize);
    fs::File::open(querypath.clone())?.read_to_end(&mut str)?;

    if load_disk {
        println!("LOAD DISK");
        let text = filebuffer::FileBuffer::open(fpath)?;
        let table = filebuffer::FileBuffer::open(format!("{}.table.bin", fpath))?;

        assert!(size_table % size_text == 0);
        let size_width = size_table / size_text;

        let mut off = 0;
        while off < str.len() {
            let length = u32::from_le_bytes(str[off..off + 4].try_into().expect("?")) as usize;
            off += 4;

            let occurances = count_occurances(
                &text,
                size_text,
                &table,
                size_table,
                &str[off..off + length],
                size_width as usize,
                false,
            );
            off += length;
            println!("Number of times present: {}", occurances);
        }
    } else {
        let mut text = Vec::with_capacity(size_text as usize);
        fs::File::open(format!("{}", fpath))?.read_to_end(&mut text)?;

        let mut table = Vec::with_capacity(size_table as usize);
        fs::File::open(format!("{}.table.bin", fpath))?.read_to_end(&mut table)?;

        assert!(size_table % size_text == 0);
        let size_width = size_table / size_text;

        let mut off = 0;
        while off < str.len() {
            let length = u32::from_le_bytes(str[off..off + 4].try_into().expect("?")) as usize;
            off += 4;

            let occurances = count_occurances_memory(
                &text,
                size_text,
                &table,
                size_table,
                &str[off..off + length],
                size_width as usize,
                false,
            );
            off += length;
            println!("Number of times present: {}", occurances);
        }
    }
    Ok(())
}

/*
 * A little bit of state for the merge operation below.
 * - suffix is suffix of one of the parts of the dataset we're merging;
this is the value we're sorting on
 * - position is the location of this suffix (so suffix = array[position..])
 * - table_index says which suffix array this suffix is a part of
 */
#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeState<'a> {
    suffix: &'a [u16],
    position: u64,
    table_index: usize,
}

impl<'a> Ord for MergeState<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.suffix.cmp(&self.suffix)
    }
}

impl<'a> PartialOrd for MergeState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/*
 * Merge together M different suffix arrays (probably created with make-part).
 * That is, given strings S_i and suffix arrays A_i compute the suffix array
 * A* = make-suffix-array(concat S_i)
 * In order to do this we just implement mergesort's Merge operation on each
 * of the arrays A_i to construct a sorted array A*.
 *
 * This algorithm is *NOT A LINEAR TIME ALGORITHM* in the worst case. If you run
 * it on a dataset consisting entirely of the character A it will be quadratic.
 * Fortunately for us, language model datasets typically don't just repeat the same
 * character a hundred million times in a row. So in practice, it's linear time.
 *
 * There are thre complications here.
 *
 * As with selfsimilar_parallel, we can't fit all A_i into memory at once, and
 * we want to make things fast and so parallelize our execution. So we do the
 * same tricks as before to make things work.
 *
 * However we have one more problem. In order to know how to merge the final
 * few bytes of array S_0 into their correct, we need to know what bytes come next.
 * So in practice we make sure that S_{i}[-HACKSIZE:] === S_{i+1}[:HACKSIZE].
 * As long as HACKSIZE is longer than the longest potential match, everything
 * will work out correctly. (I did call it hacksize after all.....)
 * In practice this works. It may not for your use case if there are long duplicates.
 */
fn cmd_merge(
    data_files: &Vec<String>,
    output_file: &String,
    num_threads: i64,
) -> std::io::Result<()> {
    // This value is declared here, but also in scripts/make_suffix_array.py
    // If you want to change it, it needs to be changed in both places.
    const HACKSIZE: usize = 100000;

    let nn: usize = data_files.len();

    // Start out by loading the data files and suffix arrays.
    let bytes: Result<Vec<_>, _> = data_files.iter().map(fs::read).collect();
    let texts: Vec<&[u16]> = bytes?.into_iter().map(|x| transmute_vec(x)).collect();

    let texts_len: Vec<usize> = texts
        .iter()
        .enumerate()
        .map(|(i, x)| x.len() - (if i + 1 == texts.len() { 0 } else { HACKSIZE }))
        .collect();

    let metadatas: Result<Vec<u64>, std::io::Error> = (0..nn)
        .map(|x| {
            let meta = fs::metadata(format!("{}.table.bin", data_files[x].clone()))?;
            assert!(meta.len() % (texts[x].len() as u64) == 0);
            return Ok(meta.len());
        })
        .collect();
    let metadatas = metadatas?;

    let big_ratio = ((texts_len.iter().sum::<usize>() as f64).log2() / 8.0).ceil() as usize;
    println!("Ratio: {}", big_ratio);

    let ratio = metadatas[0] / (texts[0].len() as u64);

    fn worker(
        texts: &Vec<&[u16]>,
        starts: Vec<usize>,
        ends: Vec<usize>,
        texts_len: Vec<usize>,
        part: usize,
        output_file: String,
        data_files: Vec<String>,
        ratio: usize,
        big_ratio: usize,
    ) {
        let nn = texts.len();
        let mut tables: Vec<TableStream> = (0..nn)
            .map(|x| make_table(format!("{}.table.bin", data_files[x]), starts[x], ratio))
            .collect();

        let mut idxs: Vec<u64> = starts.iter().map(|&x| x as u64).collect();

        let delta: Vec<u64> = (0..nn)
            .map(|x| {
                let pref: Vec<u64> = texts[..x].iter().map(|y| y.len() as u64).collect();
                pref.iter().sum::<u64>() - (HACKSIZE * x) as u64
            })
            .collect();

        let mut next_table = std::io::BufWriter::new(
            File::create(format!("{}.table.bin.{:04}", output_file.clone(), part)).unwrap(),
        );

        fn get_next_maybe_skip(
            mut tablestream: &mut TableStream,
            index: &mut u64,
            thresh: usize,
        ) -> u64 {
            //println!("{}", *index);
            let mut location = get_next_pointer_from_table_canfail(&mut tablestream);
            if location == u64::MAX {
                return location;
            }
            *index += 1;
            while location >= thresh as u64 {
                location = get_next_pointer_from_table_canfail(&mut tablestream);
                if location == u64::MAX {
                    return location;
                }
                *index += 1;
            }
            return location;
        }

        let mut heap = BinaryHeap::new();

        for x in 0..nn {
            let position = get_next_maybe_skip(&mut tables[x], &mut idxs[x], texts_len[x]);
            //println!("{} @ {}", position, x);
            heap.push(MergeState {
                suffix: &texts[x][position as usize..],
                position: position,
                table_index: x,
            });
        }

        // Our algorithm is not linear time if there are really long duplicates
        // found in the merge process. If this happens we'll warn once.
        let mut did_warn_long_sequences = false;

        let mut prev = &texts[0][0..];
        while let Some(MergeState {
            suffix: _suffix,
            position,
            table_index,
        }) = heap.pop()
        {
            //next_table.write_all(&(position + delta[table_index] as u64).to_le_bytes()).expect("Write OK");
            next_table
                .write_all(&(position + delta[table_index] as u64).to_le_bytes()[..big_ratio])
                .expect("Write OK");

            let position = get_next_maybe_skip(
                &mut tables[table_index],
                &mut idxs[table_index],
                texts_len[table_index],
            );
            if position == u64::MAX {
                continue;
            }

            if idxs[table_index] <= ends[table_index] as u64 {
                let next = &texts[table_index][position as usize..];
                //println!("  {:?}", &next[..std::cmp::min(10, next.len())]);

                let match_len = (0..50000000)
                    .find(|&j| !(j < next.len() && j < prev.len() && next[j] == prev[j]));
                if !did_warn_long_sequences {
                    if let Some(match_len_) = match_len {
                        if match_len_ > 5000000 {
                            println!("There is a match longer than 50,000,000 bytes.");
                            println!("You probably don't want to be using this code on this dataset---it's (possibly) quadratic runtime now.");
                            did_warn_long_sequences = true;
                        }
                    } else {
                        println!("There is a match longer than 50,000,000 bytes.");
                        println!("You probably don't want to be using this code on this dataset---it's quadratic runtime now.");
                        did_warn_long_sequences = true;
                    }
                }

                heap.push(MergeState {
                    suffix: &texts[table_index][position as usize..],
                    position: position,
                    table_index: table_index,
                });
                prev = next;
            }
        }
    }

    // Make sure we have enough space to take strided offsets for multiple threads
    // This should be an over-approximation, and starts allowing new threads at 1k of data
    //let num_threads = std::cmp::min(num_threads, std::cmp::max((texts.len() as i64 - 1024)/10, 1));
    println!("AA {}", num_threads);

    // Start a bunch of jobs that each work on non-overlapping regions of the final resulting suffix array
    // Each job is going to look at all of the partial suffix arrays to take the relavent slice.
    let _answer = thread::scope(|scope| {
        let mut tables: Vec<BufReader<File>> = (0..nn)
            .map(|x| {
                std::io::BufReader::new(
                    fs::File::open(format!("{}.table.bin", data_files[x])).unwrap(),
                )
            })
            .collect();

        let mut starts = vec![0; nn];

        for i in 0..num_threads as usize {
            let mut ends: Vec<usize> = vec![0; nn];
            if i < num_threads as usize - 1 {
                ends[0] =
                    (texts[0].len() + (num_threads as usize)) / (num_threads as usize) * (i + 1);
                let end_seq = &texts[0][table_load_disk(&mut tables[0], ends[0], ratio as usize)..];

                for j in 1..ends.len() {
                    ends[j] = off_disk_position(texts[j], &mut tables[j], end_seq, ratio as usize);
                }
            } else {
                for j in 0..ends.len() {
                    ends[j] = texts[j].len();
                }
            }

            for j in 0..ends.len() {
                let l = &texts[j][table_load_disk(&mut tables[j], starts[j], ratio as usize)..];
                let l = &l[..std::cmp::min(l.len(), 20)];
                println!("Text{} {:?}", j, l);
            }

            println!("Spawn {}: {:?} {:?}", i, starts, ends);

            let starts2 = starts.clone();
            let ends2 = ends.clone();
            //println!("OK {} {}", starts2, ends2);
            let texts_len2 = texts_len.clone();
            let texts2 = texts.clone();

            let _one_result = scope.spawn(move || {
                worker(
                    &texts2,
                    starts2,
                    ends2,
                    texts_len2,
                    i,
                    (*output_file).clone(),
                    (*data_files).clone(),
                    ratio as usize,
                    big_ratio as usize,
                );
            });

            for j in 0..ends.len() {
                starts[j] = ends[j];
            }
        }
    });

    println!("Finish writing");
    let mut buffer = File::create(output_file)?;
    for i in 0..texts.len() - 1 {
        buffer.write_all(transmute_slice(&texts[i][..texts[i].len() - HACKSIZE]))?;
    }
    buffer.write_all(transmute_slice(texts[texts.len() - 1]))?;
    Ok(())
}

/*
 * Given the output of either self-similar or across-similar,
 * compute byte ranges that are duplicates.
 *
 * The similar outputs are just byte values
 * [A_0, A_1, ..., A_N]
 * meaning that the bytes from (A_i, A_i + length_threshold) are duplicated somewhere.
 *
 * This script converts this to ranges [a, b) for complete ranges that should be removed.
 * For example if we have a long duplicate sequence
 *    abcdefg
 * then we might have a match for `abcde` and `bcdef` and `cdefg`
 * So instead of just saying tokens 0, 1, and 2 match, here we say that [0, 7) match.
 *
 * To do this we
 *   (a) sort the output lists, and then
 *   (b) collapse overlapping buckets.
 *
 * Note that as a result of doing this, we might have a sequence `qwerty` where the
 * entire sequence is never repeated in the dataset multiple times, but each byte
 * in the sequence is part of some length_threshold duplicate.
 */
fn cmd_collect(
    data_file: &String,
    cache_dir: &String,
    length_threshold: u64,
) -> std::io::Result<()> {
    let paths = fs::read_dir(cache_dir)?;

    let metadata_text = fs::metadata(format!("{}", data_file))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", data_file))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    assert!(size_table % size_text == 0);
    let size_width = size_table / size_text;

    let ds_name = data_file.split("/").last().unwrap();

    let mut path_list = Vec::with_capacity(1000);
    for path in paths {
        let path = path?.path().as_path().to_str().unwrap().to_string();
        if !path.starts_with(
            &Path::new(cache_dir)
                .join(format!("dups_{}_", ds_name))
                .into_os_string()
                .into_string()
                .unwrap(),
        ) {
            continue;
        }
        path_list.push(path);
    }

    // 1. Perform an initial sort of each of the found duplicates

    let outputs: Vec<Vec<u64>> = path_list.into_par_iter().map(|path| {
        let mut all_items = from_bytes(fs::read(path.clone()).unwrap(), size_width as usize);
        // let mut all_items: Vec<u64> = all_items.into_iter().filter(|&x| x % 2 == 0).collect();
        all_items.sort_unstable();
        return all_items;
    }).collect();

    let mut all_items: Vec<u64> = Vec::new();
    println!("Merging.");

    // 2. Perform a merge of the now-sorted lists

    let mut heap = BinaryHeap::new();

    // Seed the heap with the first element of each
    for (i, output) in outputs.iter().enumerate() {
        if output.len() > 0 {
            heap.push(Reverse((output[0], 0, i)));
        }
    }

    let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(1000);
    let mut prev_start;
    let mut prev_end;

    // Unroll first iteration of the loop for performance
    if let Some(Reverse((data_pointer, index, which_array))) = heap.pop() {
        prev_start = data_pointer;
        prev_end = data_pointer + length_threshold;
        // ensure this bucket has enough data to push the item
        if index + 1 < outputs[which_array].len() {
            heap.push(Reverse((
                outputs[which_array][index + 1],
                index + 1,
                which_array,
            )));
        }
    } else {
        println!(
            "No duplicates found! Either the dataset is duplicate-free or something went wrong."
        );
        return Ok(());
    }

    // Now walk the the rest of the merging
    while let Some(Reverse((data_pointer, index, which_array))) = heap.pop() {
        all_items.push(data_pointer);

        if data_pointer <= prev_end {
            prev_end = data_pointer + length_threshold;
        } else {
            ranges.push((prev_start, prev_end));
            prev_start = data_pointer;
            prev_end = data_pointer + length_threshold;
        }

        // If this array has more data, consume it
        if index + 1 < outputs[which_array].len() {
            heap.push(Reverse((
                outputs[which_array][index + 1],
                index + 1,
                which_array,
            )));
        }
    }
    ranges.push((prev_start, prev_end));

    let strout: Vec<String> = ranges.iter().map(|&x| format!("{} {}", x.0, x.1)).collect();
    println!("out\n{}", strout.join("\n"));
    Ok(())
}

/*
fn cmd_collect(data_file: &String, cache_dir: &String, length_threshold: u64)  -> std::io::Result<()> {
    let paths = fs::read_dir(cache_dir)?;


    let metadata_text = fs::metadata(format!("{}", data_file))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", data_file))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    let mut output = bitbox!(u32, Msb0; 0; metadata_text.len() as usize);

    let mut ranges:Vec<(u64,u64)> = Vec::with_capacity(1000);

    assert!(size_table % size_text == 0);
    let size_width = size_table / size_text;

    let ds_name = data_file.split("/").last().unwrap();


    let mut path_list = Vec::with_capacity(1000);
    for path in paths {
        let path = path?.path().as_path().to_str()?.to_string();
        if !path.starts_with(&Path::new(cache_dir).join(format!("dups_{}_", ds_name.clone())).into_os_string().into_string()?) {
            continue;
        }
        path_list.push(path);
    }

    // 1. Perform an initial sort of each of the found duplicates

    let mut result = Vec::with_capacity(100);
    thread::scope(|scope| {
        for path in path_list.into_iter() {
            let path = path.clone();
            let out = scope.spawn(move || {
                let all_items = from_bytes(fs::read(path.clone())?, size_width as usize);
        for x in all_items {
            for y in 0..length_threshold {
                    output.set_unchecked((x+y) as usize, true);
                }
        }
        return 0;
            });
            result.push(out);
        }
    });
    let _outputs:Vec<u64> = result.into_iter().map(|t| t.join()).collect();
    Ok(())
}
*/

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    match &args.command {
        Commands::Make { data_file } => {
            cmd_make(data_file)?;
        }

        Commands::MakePart {
            data_file,
            start_byte,
            end_byte,
        } => {
            cmd_make_part(data_file, *start_byte as u64, *end_byte as u64)?;
        }

        Commands::CountOccurrences {
            data_file,
            query_file,
            print_location,
            load_disk,
        } => {
            cmd_count_occurrences(data_file, query_file, *print_location, *load_disk)?;
        }

        Commands::CountOccurrencesMulti {
            data_file,
            query_file,
            load_disk,
        } => {
            cmd_count_occurrences_multi(data_file, query_file, *load_disk)?;
        }

        Commands::Merge {
            suffix_path,
            output_file,
            num_threads,
        } => {
            cmd_merge(suffix_path, output_file, *num_threads)?;
        }

        Commands::Collect {
            data_file,
            cache_dir,
            length_threshold,
        } => {
            cmd_collect(data_file, cache_dir, *length_threshold)?;
        }
    }

    Ok(())
}
