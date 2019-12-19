# BRSNMF
Matlab codes for Detecting Drug Communities and Predicting Comprehensive Drug-Drug Interactions via Balance Regularized Semi-Nonnegative Matrix Factorization

## Data: 

######'DDI.mat' contains the following variables 


|Name                      |Size                 |Bytes  |Class     |Attributes|          
| ------------ | ------------ | ------------ | ------------ | ------------ |          
|  Adj_DCA                |1601x47                |601976  |double    |drug-binding proteins: carriers, non-targets|          
|  Adj_DEN                |1601x230              |2945840  |double    |drug-binding proteins: enzymes, non-targets|          
|  Adj_DTI                |1601x1213            |15536104  |double    |drug-binding proteins: targets|          
|  Adj_DTR                |1601x152              |1946816  |double    |drug-binding proteins: transporters, non-targets|           
|  Adj_V4                 |1562x1562            |19518752  |double    |the DDI matrix containing +1 (enhancive), -1 (degressive) and 0|          
|  DrugId_V4              |1562x1                  |12496  |double    |the indices of drug names (V4) in DrugNames|          
|  DrugId_V5                |39x1                    |312  |double    |the indices of drug names (V5) in DrugNames|           
|  DrugNames              |1601x1                 |201726  |cell      |all the drug names sorted by alphabet|          
|  DrugNames_V4           |1562x1                 |196812  |cell      |all the drug names in V4 sorted by alphabet|          
|  DrugNames_V5             |39x1                   |4914  |cell      |all the drug names in V5 sorted by alphabet|          
|  structure_feature      |1601x881             |11283848  |double    |the PubChem fingerprints|          
---------------------

###### 'DCA_1601.mat', 'DEN_1601.mat', 'DTI_1601.mat', 'DTR_1601.mat' contain four variables. For example, DTI_1601.mat contains 

|Name                      |Size                 |Bytes  |Class     |Attributes|
| ------------ | ------------ | ------------ | ------------ | ------------ |
|  Adj_DTI          |1601x1213            |15536104  |double    |drug-binding proteins: targets|          
|  DTI              |4996x3                |1900764  |cell      |DBP entries|          
|  DTI_name         |1352x1                 |170352  |cell      |drugs having at least one target|          
|  Target_name      |1213x1                 |150420  |cell      |the names of drug targets|

The contents of other files are organized in a similar way.

## Codes: are being cleaned.
