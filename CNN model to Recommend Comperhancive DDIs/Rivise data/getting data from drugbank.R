install.packages("devtools")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("AnnotationDbi")

source("http://bioconductor.org/biocLite.R")
biocLite("org.Hs.eg.db")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("org.Hs.eg.db")

devtools::install_github("yduan004/drugbankR")


library(drugbankR)

drugbank_dataframe <- dbxml2df(xmlfile="drugbank.xml")
# , version="5.1.3"
df2SQLite(dbdf=drugbank_dataframe)
# , version="5.1.3"

install.packages("openxlsx", dependencies=TRUE)

library(openxlsx)

# # read data from an Excel file or Workbook object into a data.frame
# df <- read.xlsx('name-of-your-excel-file.xlsx')

# for writing a data.frame or list of data.frames to an xlsx file
write.xlsx(drugbank_dataframe, 'drugbank.xlsx')



all <- queryDB(type = "getAll", db_path="drugbank_5.1.3.db") # get the entire drugbank dataframe
dim(all)
ids <- queryDB(type = "getIDs", db_path="drugbank_5.1.3.db") # get all the drugbank ids
ids[1:4]

# given drugbank ids, determine whether they are FDA approved
queryDB(ids = c("DB00001","DB00002"),type = "whichFDA", db_path="drugbank_5.1.3.db") 

# given drugbank ids, get their targets
queryDB(ids = c("DB00001","DB00002"),type = "getTargets", db_path="drugbank_5.1.3.db") 
