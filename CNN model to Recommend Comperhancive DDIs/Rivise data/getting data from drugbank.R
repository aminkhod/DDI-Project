install.packages("devtools")
devtools::install_github("yduan004/drugbankR")


library(drugbankR)

drugbank_dataframe <- dbxml2df(xmlfile="drugbank.xml") 
df2SQLite(dbdf=drugbank_dataframe, version="5.1.3")

all <- queryDB(type = "getAll", db_path="drugbank_5.1.3.db") # get the entire drugbank dataframe
dim(all)
ids <- queryDB(type = "getIDs", db_path="drugbank_5.1.3.db") # get all the drugbank ids
ids[1:4]

# given drugbank ids, determine whether they are FDA approved
queryDB(ids = c("DB00001","DB00002"),type = "whichFDA", db_path="drugbank_5.1.3.db") 

# given drugbank ids, get their targets
queryDB(ids = c("DB00001","DB00002"),type = "getTargets", db_path="drugbank_5.1.3.db") 
