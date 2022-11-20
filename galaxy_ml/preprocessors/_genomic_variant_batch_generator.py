
import warnings

import numpy as np

from sklearn.base import BaseEstimator

import tabix

from ._fasta_iterator import FastaToArrayIterator
from ..externals import selene_sdk


class GenomicVariantBatchGenerator(BaseEstimator):
    """`keras.utils.Sequence` capable sequence array generator
    from a reference genome and VCF (variant call format) file.

    Parameters
    ----------
    ref_genome_path : str
        File path to the reference genomce, usually in fasta
        format.
    vcf_path : str
        File path to the VCF dataset.
    blacklist_regions : str
        E.g., 'hg38'. For more info, refer to
        `selene_sdk.sequences.Genome`.
    seq_length : int, default=1000
        Retrived sequence length.
    output_reference : bool, default is False.
        If True, output reference sequence instead.
    """
    def __init__(self, ref_genome_path=None, vcf_path=None,
                 blacklist_regions='hg38', seq_length=1000,
                 output_reference=False):
        self.ref_genome_path = ref_genome_path
        self.vcf_path = vcf_path
        self.blacklist_regions = blacklist_regions
        self.seq_length = seq_length
        self.output_reference = output_reference
        self.n_bases = 4
        self.BASE_TO_INDEX = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'a': 0, 'c': 1, 'g': 2, 't': 3,
        }
        self.UNK_BASE = 'N'

    def set_processing_attrs(self):
        self.reference_genome_ = selene_sdk.sequences.Genome(
            input_path=self.ref_genome_path,
            blacklist_regions=self.blacklist_regions)

        self.start_radius_ = self.seq_length // 2
        self.end_radius_ = (self.seq_length + 1) // 2

        variants = selene_sdk.predict._variant_effect_prediction.\
            read_vcf_file(self.vcf_path)

        # store indices of variants whose ref sequence doesn't match the
        # reference genome
        self.unmatches = []

        # clean variants
        self.variants = []
        for chrom, pos, name, ref, alt, strand in variants:
            center = pos + len(ref) // 2
            start = center - self.start_radius_
            end = center + self.end_radius_

            if isinstance(self.reference_genome_,
                          selene_sdk.sequences.Genome):
                if "chr" not in chrom:
                    chrom = "chr" + chrom
                if "MT" in chrom:
                    chrom = chrom[:-1]

            if not self.reference_genome_.coords_in_bounds(chrom, start, end):
                continue
            blacklist_tabix = getattr(self.reference_genome_,
                                      '_blacklist_tabix', None)
            if blacklist_tabix:
                try:
                    found = 0
                    rows = blacklist_tabix.query(chrom, start, end)
                    for row in rows:
                        found = 1
                        break
                    if found:
                        continue
                except tabix.TabixError:
                    pass

            for al in alt.split(','):
                self.variants.append((chrom, pos, name, ref, al, strand))

        return self

    def flow(self, batch_size=32):
        """
        Parameters
        ----------
        batch_size : int, default is 32
        """
        if not hasattr(self, 'reference_genome_'):
            self.set_processing_attrs()

        X = np.arange(len(self.variants))[:, np.newaxis]
        return FastaToArrayIterator(
            X, self, y=None, batch_size=batch_size, sample_weight=None,
            shuffle=False)

    def apply_transform(self, idx):
        chrom, pos, name, ref, alt, strand = self.variants[idx]

        center = pos + len(ref) // 2
        start = center - self.start_radius_
        end = center + self.end_radius_
        ref_len = len(ref)

        if self.output_reference:   # return reference encoding
            seq_encoding = self.reference_genome_.get_encoding_from_coords(
                chrom, start, end, strand=strand)
            ref_encoding = self.reference_genome_.sequence_to_encoding(ref)

            if ref_len < self.seq_length:
                match, seq_encoding, seq_at_ref = selene_sdk.predict.\
                    _variant_effect_prediction._handle_standard_ref(
                        ref_encoding,
                        seq_encoding,
                        self.start_radius_,
                        self.reference_genome_)
            else:
                match, seq_encoding, seq_at_ref = selene_sdk.predict.\
                    _variant_effect_prediction._handle_long_ref(
                        ref_encoding,
                        seq_encoding,
                        self.start_radius_,
                        self.end_radius_,
                        self.reference_genome_)
            if not match:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}), "
                              "reference does not match the reference genome. "
                              "Reference genome contains {5} instead. ".format(
                                  chrom, pos, name, ref, alt, seq_at_ref))
                self.unmatches.append(idx)
            return seq_encoding

        # return variant encoding
        if alt == '*':
            alt = ''
        alt_len = len(alt)
        sequence = None

        if alt_len > self.seq_length:
            sequence = selene_sdk.predict._common._truncate_sequence(
                alt, self.seq_length)
        elif ref_len == alt_len:  # substitution
            sequence = self.reference_genome_.get_sequence_from_coords(
                chrom, start, end, strand=strand)
            remove_ref_start = self.start_radius_ - ref_len // 2 - 1
            sequence = (
                sequence[:remove_ref_start]
                + alt
                + sequence[remove_ref_start + ref_len:]
            )
        else:  # insertion or deletion
            seq_lhs = self.reference_genome_.get_sequence_from_coords(
                chrom,
                pos - 1 - self.start_radius_ + alt_len // 2,
                pos - 1,
                strand=strand,
                pad=True,
            )
            seq_rhs = self.reference_genome_.get_sequence_from_coords(
                chrom,
                pos - 1 + ref_len,
                pos - 1 + ref_len + self.end_radius_ - (alt_len + 1) // 2,
                strand=strand,
                pad=True,
            )
            sequence = seq_lhs + alt + seq_rhs
        alt_encoding = self.reference_genome_.sequence_to_encoding(
            sequence)

        return alt_encoding

    def close(self):
        if hasattr(self, 'reference_genome_'):
            self.reference_genome_.genome.close()
