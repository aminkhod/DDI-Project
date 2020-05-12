## User-friendly extensions of the DrugBank database

[![DOI 10.5281/zenodo.45579](https://zenodo.org/badge/doi/10.5281/zenodo.45579.svg)](https://doi.org/10.5281/zenodo.45579)

[DrugBank](http://www.drugbank.ca/) is a publicly-available resource of drug information [[1](https://doi.org/10.1093/nar/gkt1068)]. We rely on DrugBank for our [project to repurpose drugs](https://doi.org/10.15363/thinklab.4 "Thinklab: Repurposing drugs on a hetnet"). We are conducting this project openly on ThinkLab, and this README will reference Thinklab discussions providing greater detail.

This repository contains several code and data components:

+ `parse.ipynb` -- extracts information from the DrugBank xml download into a [tsv file](data/drugbank.tsv) where each row represents a drug. A [subset](data/drugbank-slim.tsv) referred to as slim contains only drugs that are approved, small molecules, and contain an InChI structure ([discussion](https://doi.org/10.15363/thinklab.d70#192)). We also extract the [interacting proteins](data/proteins.tsv) for each drug, which include targets, enzymes, transporters, and carriers ([dicussion](https://doi.org/10.15363/thinklab.d65)).

+ `similarity.ipynb` -- calculates chemical similarity between drugbank compounds using extended connectivity fingerprints ([dicussion](http://thinklab.com/d/70)). Similarities range from 0 to 1. The full similarity download is [available on figshare](https://doi.org/10.6084/m9.figshare.1418386). The subset of similarities for slim compounds is [on github](data/similarity-slim.tsv.gz).

+ `unichem-map.ipynb` -- maps DrugBank compounds to 30 other compound resources using [UniChem](http://www.ebi.ac.uk/unichem/info/widesearchInfo). The mapping is based on atomic connectivity and ignores differences in small molecular details. Mappings are available in a [bulk download](data/mapping.tsv.gz) or for [individual resources](data/mapping). Summary statistics are also [available](data/mapping-counts.tsv) ([discussion](http://thinklab.com/d/70)).

+ `pubchem-map.ipynb` -- DrugBank compounds were mapped to [PubChem](https://pubchem.ncbi.nlm.nih.gov/search/) based on exact InChi string matches. The mapping is available as a [tsv file](data/pubchem-mapping.tsv).

+ `parse-halflife.ipynb` -- extracts half-life and other structural information from the Drugbank xml download into a [tsv file](data/drugbank_halflife.tsv) where each row represents a drug. The half-life information was listed as free text in Drugbank. We manually extract the numeric value from free text into a [xlsx file](data/drugbank_halflife_curated.xlsx). All values were converted to hours. If the value was listed as time range (e.g. a ~ b) in DrugBank, average was calculated (e.g. (a + b)/2).

+ `extract-curated-halflife.ipynb` -- extracts subset of drugs with curated half-life into a [tsv file](data/drugbank_subset_halflife_curated.tsv) where each row represents a drug.

+ `predict-halflife.ipynb` -- builds supervised learning models to predict half-life based on structural properties of drugs.

## License

DrugBank content and derivates are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/ "Creative Commons Attribution-NonCommercial 4.0 International"). Original content is released as [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/ "CC0 1.0 Universal: Public Domain Dedication")
