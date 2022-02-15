import sys
from tqdm import tqdm
import pandas as pd
import pymongo

# Connect to luima mongodb
# Input: dictionary with keys DB_HOST, DB_PORT, DB_DATABASE, DB_USER, DB_PASSWORD
def connect_db(db_details):
    db = pymongo.MongoClient(host=db_details["DB_HOST"],
                             port=db_details["DB_PORT"])[db_details["DB_DATABASE"]]
    db.authenticate(db_details["DB_USER"], db_details["DB_PASSWORD"])
    return db


# Load the lists of case ids, making sure that there are no duplicates
def load_case_ids(fnames):
    id_set = set()
    l = []
    for fname in fnames:
        with open(fname, "r") as f:
            case_ids = f.read().splitlines()
        new_ids = list(set(case_ids) - id_set)
        l.append(new_ids)
        id_set.update(new_ids)
    print(f"Loaded {', '.join([str(len(x)) for x in l])} documents.")
    return l


# Get the balance metadata class label
def get_metadata_class(progcode, isscode, isslev1, diagcode):                                                                                 
    if progcode != 2:                                                
        return 0                                      
    else:                              
        if isscode == 8:                                                                   
            return 2                                            
        elif isscode == 9:
            return 3
        elif isscode == 17:
            return 4
        elif isscode == 12:
            if isslev1 != 4:
                return 11
            elif isslev1 == 4:
                if diagcode == 5:
                    return 12
                elif diagcode == 6:
                    return 13
                elif diagcode == 7:
                    return 14
                elif diagcode == 8:
                    return 15
                elif diagcode == 9:
                    return 16
        elif isscode == 15:
            if isslev1 != 3:
                return 5
            elif isslev1 == 3:
                if diagcode == 5:
                    return 6
                if diagcode == 6:
                    return 7
                if diagcode == 7:
                    return 8
                if diagcode == 8:
                    return 9
                if diagcode == 9:
                    return 10
        else:
            return 1
    assert False, "Failed to assign metadata class to this example"


class MetadataBuilder():

    def __init__(self, db, metadata_fpath, case_ids):
        self.db = db
        self.metadata_fpath = metadata_fpath
        self.case_ids = case_ids
        self.metadata_dict = {}

    # Convert a column to integer
    def convert_to_int(self, col):
        return [int(x) if x != "NA" else -1 for x in col]

    # Get the 1st digit of the diagnostic code from isslev2
    def get_diag_code(self, isslev2):
        return [int(str(x)[0]) if x != "NA" else -1 for x in isslev2]

    def build_metadata(self):
        print("Creating Metadata dictionary...")
        df_meta = pd.DataFrame(list(self.db.appeals_meta_wscraped.find(
                               {"tiread2": {"$in": self.case_ids}},
                               {"_id": 0})))
        df_meta["isscode"] = self.convert_to_int(df_meta["isscode"])
        df_meta["issprog"] = self.convert_to_int(df_meta["issprog"])
        df_meta["isslev1"] = self.convert_to_int(df_meta["isslev1"])
        df_meta["diagcode"] = self.get_diag_code(df_meta["isslev2"])
        df_meta["year"] = df_meta["imgadtm"].str[0:4].astype(int)
        df_meta["bva_id"] = df_meta["tiread2"]
        df_meta["class"] = df_meta.apply(lambda r: 
                            get_metadata_class(r["issprog"], r["isscode"],
                                r["isslev1"], r["diagcode"]), axis=1)
        df_meta = df_meta[["bva_id", "year", "class"]]
        df_meta.drop_duplicates(subset="bva_id", keep='first', inplace=True)
        df_meta.to_csv(self.metadata_fpath, index=False)
        print(f"Saved {df_meta.shape[0]} rows of metadata.")


if __name__ == "__main__":
    # metadata_fpath: location to store metadata
    # id_list: locations of the respective case ids
    metadata_fpath = './metadata.csv'
    id_list = ['../data/train_data_ids.txt',
               '../data/dev_data_ids.txt']
    train_ids, dev_ids = load_case_ids(id_list)

    # Connect to DB
    # Store credentials in a db_config.py like so:
    #       db_details = {
    #          "DB_HOST"    : "XXX",
    #          "DB_PORT"    : XXX,
    #          "DB_DATABASE": "XXX",
    #          "DB_USER"    : "XXX",
    #          "DB_PASSWORD": "XXX"
    #          }

    from db_config import db_details
    db = connect_db(db_details)
    metadata_builder = MetadataBuilder(db, metadata_fpath, train_ids+dev_ids)
    metadata_builder.build_metadata()


