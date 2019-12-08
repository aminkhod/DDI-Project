PredictByMonopartite.m is the top-level calling function
PredictS2_By_MatrixFactorization.m is an experimental function
nmf.m和semi_nmf.m is experimental method function
EstimationAUC.m和prec_rec.m is calculate AUC, AUPR function

Example:
When using nmf in Binary DDI network.
PredictByMonopartite(DDI_binary,feature,CV,nComp,'Binary','S2');

When using semi-nmf in Comprehensive DDI network.
PredictByMonopartite(DDI_triple,feature,CV,nComp,'Triple','S2');

CV is cross-validation，nComp is the potential space dimension.
The most important is set nmf and semi_nmf in PredictS2_By_MatrixFactorization.