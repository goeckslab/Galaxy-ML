from pathlib import Path

import pytest


@pytest.fixture
def genomic_files(tmp_path):
    """Small real genome inputs; no external reference genome or downloads."""
    reference = tmp_path / 'reference.fa'
    reference.write_text('>chr1\n' + 'ACGT' * 512 + '\n>chr2\n'
                         + 'TGCA' * 512 + '\n')
    intervals = tmp_path / 'intervals.bed'
    intervals.write_text(''.join(
        f'chr{i % 2 + 1}\t{100 + i * 100}\t{120 + i * 104}\n'
        for i in range(12)))
    variants = tmp_path / 'variants.vcf'
    variants.write_text(
        '##fileformat=VCFv4.2\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
        '1\t101\tsnv1\tA\tG\t.\tPASS\t.\n'
        '1\t202\tsnv2\tC\tT\t.\tPASS\t.\n'
        '1\t301\tmulti\tA\tC,T\t.\tPASS\t.\n')
    return dict(ref_genome_path=str(reference), intervals_path=str(intervals),
                target_path=str(Path(__file__).parent / 'data'
                                / 'genomic_targets.bed.gz'),
                vcf_path=str(variants))


@pytest.fixture
def genomic_generator(genomic_files):
    from galaxy_ml.preprocessors import GenomicIntervalBatchGenerator

    generator = GenomicIntervalBatchGenerator(
        **{key: genomic_files[key] for key in
           ('ref_genome_path', 'intervals_path', 'target_path')},
        features=['binding'], blacklist_regions=None, seq_length=32,
        center_bin_to_predict=8, shuffle=False, seed=42, random_state=0)
    yield generator
    generator.close()
