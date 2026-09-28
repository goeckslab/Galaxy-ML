`genomic_targets.bed` labels chr1 as positive and leaves chr2 negative in
the synthetic genome created by `genomic_files`. The compressed file and
index are checked in so tests need only the existing pytabix dependency.
Regenerate them with HTSlib:

```sh
bgzip -c genomic_targets.bed > genomic_targets.bed.gz
tabix -f -p bed genomic_targets.bed.gz
```
