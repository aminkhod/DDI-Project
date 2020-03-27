Reference:
Shi, J.-Y., Huang, H, Li J.-X., Lei P., Zhang, Y.-N., Yiu, S-M. Predicting Com-prehensive Drug-Drug Interactions for New Drugs via Triple Matrix Factoriza-tion. In: Rojas I., Ortu?o F. (eds) Bioinformatics and Biomedical Engineering. IWBBIO 2017. Lecture Notes in Computer Science, Springer. vol 10208. 2017. p.108-117

The extension of the reference paper was invited to submit to BMC Bioinformatics.

The variables in the mat file

  Name                       Size                 Bytes  Class     Attributes

  DDI_clean                603x603              2908872  double    DDI interaction matrix( +1 enhancive, -1 degressive, 0)        
  DDI_neg                  603x603              2908872  double    DDI interaction matrix( +1 enhancive,  0)          
  DDI_pos                  603x603              2908872  double    DDI interaction matrix( +1 degressive,  0)          
  DrugNames_clean          603x1                  75978  cell      Drug Names          
  OffsideName             9149x1                1408206  cell      Names of side effects in OFFSIDES          
  Offsides_PCA_clean       603x563              2715912  double    OFFSIDES profiles by PCA          
  Offsides_clean           603x9149            44134776  double    Original OFFSIDES profiles          
  Sim_Jaccard              603x603              2908872  double    Jaccard similarity of OFFSIDES profiles          
  degree_                    1x603                 4824  double    degreee of drugs in comprehensive DDIs           
  degree_neg                 1x603                 4824  double    degreee of drugs in degressive DDIs          
  degree_pos                 1x603                 4824  double    degreee of drugs in enhancive DDIs          
