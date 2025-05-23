import pandas as pd
import numpy as np
import argparse 

from create_tace_df import create_tace_df
from add_text import add_edu_text

# Using Tace matches

def tace_matches(all_df):

    # Interchangeable relations
    nca_match = all_df.loc[all_df['Agreement'] == 'NCA']

    # NNNS
    # nuclearity switched v1
    # Matching: “C1=A2 and A1=C2”
    # Disagreement: “N/N-N/S, ≠R”
    nnns_match_prelim = all_df.loc[all_df['Disagreement'] == "N/N-N/S, ≠R"]
    # The category is not just when this is in disagreement, but when in the "Matching" column we have: C1=C2 and A1=A2, C1=A2 and A1=C2
    nnns_match = nnns_match_prelim.loc[(nnns_match_prelim['Matching'] == "C1=C2 and A1=A2") | (nnns_match_prelim['Matching'] == "C1=A2 and A1=C2")]

    # nuclearity switched v2
    # Matching: “C1=A2 and A1=C2”
    # Disagreement: “N/S, ≠R”
    nnns_match_prelim2 = all_df.loc[all_df['Disagreement'] == "N/S, ≠R"]
    # The category is not just when this is in disagreement, but when in the "Matching" column we have: C1=A2 and A1=C2
    nnns_match2 = nnns_match_prelim2.loc[nnns_match_prelim2['Matching'] == "C1=A2 and A1=C2"]

    nnns_all_matches = pd.concat([nnns_match,nnns_match2])

    #Perfect match
    # NRCA
    nrca_match = all_df.loc[all_df['Agreement'] == 'NRCA']

    tace_categories = pd.concat([nrca_match, nca_match, nnns_all_matches])
    others = all_df.drop(tace_categories.index)

    return tace_categories, others, nrca_match, nca_match, nnns_all_matches

# First step is to rename columns to left/right
#A is left, C is right on both sides
def rename_columns(add_no_matches):
    v1 = add_no_matches.query("`A1-A` <= `C2-A` & `A1-B` <= `C2-B`")
    v1 = v1.assign(left_outer_A=v1['A1-A'],left_inner_A=v1['A2-A'],right_inner_A=v1['C1-A'],right_outer_A=v1['C2-A'],left_outer_B=v1['A1-B'],
              left_inner_B=v1['A2-B'],right_inner_B=v1['C1-B'],right_outer_B=v1['C2-B'])
    #C is left, A is right on both sides
    v2 = add_no_matches.query("`C1-A` <= `A2-A` & `C1-B` <= `A2-B`")
    v2 = v2.assign(right_inner_A=v2['A1-A'],right_outer_A=v2['A2-A'],left_outer_A=v2['C1-A'],left_inner_A=v2['C2-A'],
           right_inner_B=v2['A1-B'],right_outer_B=v2['A2-B'],left_outer_B=v2['C1-B'],left_inner_B=v2['C2-B'])

    #C is left in A, C is right in B
    v3 = add_no_matches.query("`C1-A` <= `A2-A` & `A1-B` <= `C2-B`")
    v3 = v3.assign(right_inner_A=v3['A1-A'],right_outer_A=v3['A2-A'],left_outer_A=v3['C1-A'],left_inner_A=v3['C2-A'],
         left_outer_B=v3['A1-B'],left_inner_B=v3['A2-B'],right_inner_B=v3['C1-B'],right_outer_B=v3['C2-B'])

    #C is right in A, C is left in B
    v4 = add_no_matches.query("`A1-A` <= `C2-A` & `C1-B` <= `A2-B`")
    v4 = v4.assign(left_outer_A=v4['A1-A'],left_inner_A=v4['A2-A'],right_inner_A=v4['C1-A'],right_outer_A=v4['C2-A'],right_inner_B=v4['A1-B'],
     right_outer_B=v4['A2-B'],left_outer_B=v4['C1-B'],left_inner_B=v4['C2-B'])

    return pd.concat([v1,v2,v3,v4])

# Function for merging the two new rows
# Providing a list of indices to remove from the no matches
def merge_clean_drop(lhs, rhs, nomatch):

    assert len(lhs) == len(rhs)

    q = pd.DataFrame()

    for i in range(len(lhs)):

        if lhs[i] != rhs[i]:
            df1 = nomatch[nomatch["Unnamed: 0"] == lhs[i]]
            df2 = nomatch[nomatch["Unnamed: 0"] == rhs[i]]

            m = df1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,31,33,34]]
            n = df2.iloc[:,[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,35,36]]
            o = pd.concat([m,n], axis="columns")
            p = pd.DataFrame(o.iloc[0].combine_first(o.iloc[1])).transpose()

            q = pd.concat([q, p])

        else:
            p = nomatch[nomatch["Unnamed: 0"] == lhs[i]]
            q = pd.concat([q, p])

    idx_to_pop = list(set(lhs + rhs))

    return q, idx_to_pop

def remove_matches(new_matches, old_matches, tace_match=False): 

    if tace_match == True:

        old_matches = old_matches.drop(new_matches.index)

        return old_matches

    else:
        left_idx = new_matches["Unnamed: 0_x"].to_list()
        right_idx = new_matches["Unnamed: 0_y"].to_list()
        new_matches_updated, poplist = merge_clean_drop(left_idx, right_idx, old_matches)
        old_matches = old_matches.drop(poplist)
        no_matches_left = old_matches[["Unnamed: 0",'C1-A','C2-A','A1-A','A2-A',"Relation-A","Nuc-A","filename"]].dropna()
        no_matches_right = old_matches[["Unnamed: 0",'C1-B','C2-B','A1-B','A2-B',"Relation-B","Nuc-B","filename"]].dropna()

        new_no_matches = pd.merge(no_matches_left, no_matches_right, on="filename")
        new_no_matches = rename_columns(new_no_matches)

        return new_matches_updated, new_no_matches, old_matches
    

# We first look at the pairs of annotations where Tace has found some kind of match
def more_tace_matches(others_renamed, tace_match=True, no_matches=None):

    # Relation mismatch
    rel_mismatch2 = others_renamed.query("`left_outer_A` == `left_outer_B` & `left_inner_A` == `left_inner_B` & `right_outer_A` == `right_outer_B` & `right_inner_A` == `right_inner_B` & `Nuc-A` == `Nuc-B`")

    if tace_match==True:
        others_renamed = remove_matches(rel_mismatch2, others_renamed, tace_match)
    if tace_match==False:
        rel_mismatch2, others_renamed, no_matches = remove_matches(rel_mismatch2, no_matches, tace_match)

    # Nuclearity switched
    nuc_switched2 = others_renamed.query("`left_outer_A` == `left_outer_B` & `left_inner_A` == `left_inner_B` & `right_outer_A` == `right_outer_B` & `right_inner_A` == `right_inner_B` & `Nuc-A` != `Nuc-B`")

    if tace_match==True:
        others_renamed = remove_matches(nuc_switched2, others_renamed, tace_match)
    if tace_match==False:
        nuc_switched2, others_renamed, no_matches = remove_matches(nuc_switched2, no_matches, tace_match)

    #Left right
    left_right_new = pd.concat([others_renamed.query("`right_outer_A` == `left_outer_B` & `right_inner_A` == `left_inner_B` & `left_outer_A` != `right_outer_B` & `left_inner_A` != `right_inner_B`"),
                            others_renamed.query("`left_outer_A` == `right_outer_B` & `left_inner_A` == `right_inner_B` & `right_outer_A` != `left_outer_B` & `right_inner_A` != `left_inner_B`")])

    if tace_match==True:
        others_renamed = remove_matches(left_right_new, others_renamed, tace_match)
    if tace_match==False:
        left_right_new, others_renamed, no_matches = remove_matches(left_right_new, no_matches, tace_match)
    
    t1 = pd.concat([others_renamed.query("`Relation-A` == `Relation-B` & `left_outer_A` == `left_outer_B` & `right_outer_A` == `right_outer_B` & `left_inner_A` != `left_inner_B`"),
                others_renamed.query("`Relation-A` == `Relation-B` & `left_outer_A` == `left_outer_B` & `right_outer_A` == `right_outer_B` & `right_inner_A` != `right_inner_B`")])
    t1.drop_duplicates(keep="first", inplace=True)

    if tace_match==True:
        others_renamed = remove_matches(t1, others_renamed, tace_match)
    if tace_match==False:
        t1, others_renamed, no_matches = remove_matches(t1, no_matches, tace_match)
    # - T2b --> same relation, same meeting point, one argument is identical
    t2b = pd.concat([others_renamed.query("`Relation-A` == `Relation-B` & `right_inner_A` == `right_inner_B` & `right_outer_A` == `right_outer_B`"),
                 others_renamed.query("`Relation-A` == `Relation-B` & `left_inner_A` == `left_inner_B` & `left_outer_A` == `left_outer_B`")])

    if tace_match==True:
        others_renamed = remove_matches(t2b, others_renamed, tace_match)
    if tace_match==False:
        t2b, others_renamed, no_matches = remove_matches(t2b, no_matches, tace_match)

    #- T2a --> same relation, same meeting point, both arguments are different sizes
    t2a = pd.concat([others_renamed.query("`Relation-A` == `Relation-B` & `left_inner_A` == `left_inner_B` & `left_outer_A` != `left_outer_B`"),
                others_renamed.query("`Relation-A` == `Relation-B` & `right_inner_A` == `right_inner_B` & `right_outer_A` != `right_outer_B`")]).drop_duplicates(keep="first")

    if tace_match==True:
        others_renamed = remove_matches(t2a, others_renamed, tace_match)
    if tace_match==False:
        t2a, others_renamed, no_matches = remove_matches(t2a, no_matches, tace_match)
        

    # T3: same relation, one of the endpoints is the same, the other isn't, meeting point is different
    t3 = pd.concat([others_renamed.query("`Relation-A` == `Relation-B` & `left_outer_A` == `left_outer_B` & `right_outer_A` != `right_outer_B` & `left_inner_A` != `left_inner_B`"),
                others_renamed.query("`Relation-A` == `Relation-B` & `left_outer_A` != `left_outer_B` & `right_outer_A` == `right_outer_B` & `left_inner_A` != `left_inner_B`")])

    if tace_match==True:
        others_renamed = remove_matches(t3, others_renamed, tace_match)
    if tace_match==False:
        t3, others_renamed, no_matches = remove_matches(t3, no_matches, tace_match)

    # t4: different relation, different split point
    t4 = pd.concat([others_renamed.query("`Relation-A` != `Relation-B` & `left_outer_A` == `left_outer_B` & `right_outer_A` == `right_outer_B` & `left_inner_A` != `left_inner_B`"),
                others_renamed.query("`Relation-A` != `Relation-B` & `left_outer_A` == `left_outer_B` & `right_outer_A` == `right_outer_B` & `right_inner_A` != `right_inner_B`")]).drop_duplicates(keep="first")

    if tace_match==True:
        others_renamed = remove_matches(t4, others_renamed, tace_match)
    if tace_match==False:
        t4, others_renamed, no_matches = remove_matches(t4, no_matches, tace_match)

    # - T5 --> different relation, same meeting point, one argument is identical
    t5 = pd.concat([others_renamed.query("`Relation-A` != `Relation-B` & `right_inner_A` == `right_inner_B` & `right_outer_A` == `right_outer_B`"),
                 others_renamed.query("`Relation-A` != `Relation-B` & `left_inner_A` == `left_inner_B` & `left_outer_A` == `left_outer_B`")])

    if tace_match==True:
        others_renamed = remove_matches(t5, others_renamed, tace_match)
    if tace_match==False:
        t5, others_renamed, no_matches = remove_matches(t5, no_matches, tace_match)

    t1["Subcategory"] = "T1"
    t2a["Subcategory"] = "T2a"
    t2b["Subcategory"] = "T2b"
    t3["Subcategory"] = "T3"
    t4["Subcategory"] = "T4"
    t5["Subcategory"] = "T5"

    scopes = pd.concat([t1,t2a,t2b,t3,t4,t5])

    return rel_mismatch2, nuc_switched2, left_right_new, scopes, others_renamed


def create_categories_pipeline(concat_tace, output_file):
    # Read CSV file (concatenated Tace file with EDU text)
    all_df = concat_tace  #pd.read_csv(open("tace_with_text.csv", "r"))
    # Replace nan, convert to floats
    all_df = all_df.replace(to_replace="blank", value=np.nan)
    all_df[[ 'C1-A','C2-A','A1-A','A2-A','C1-B','C2-B','A1-B','A2-B']] = all_df[[ 'C1-A','C2-A','A1-A','A2-A','C1-B','C2-B','A1-B','A2-B']].astype(float)

    tace_categories, others, nrca_match, nca_match, nnns_all_matches = tace_matches(all_df)
    others_matches = others[others['Matching'] != "No matching"]
    other_matches_renamed = rename_columns(others_matches)
    assert len(others_matches) == len(other_matches_renamed)

    # Now collating the non-Tace matches
    rel_mismatch2, nuc_switched2, left_right_new, scopes, other_matches_renamed1 = more_tace_matches(other_matches_renamed, tace_match=True)

    rel_mismatch = pd.concat([nca_match, rel_mismatch2])
    nuc_switched = pd.concat([nnns_all_matches, nuc_switched2])
    all_other_categories = pd.concat([nrca_match, rel_mismatch, nuc_switched, left_right_new, scopes])

    no_matches = all_df.drop(all_other_categories.index)
    # Also get rid of the misc tace matches
    no_matches = no_matches.drop(other_matches_renamed1.index)

    no_matches['Unnamed: 0'] = no_matches.index

    no_matches_left = no_matches[["Unnamed: 0",'C1-A','C2-A','A1-A','A2-A',"Relation-A","Nuc-A","filename"]].dropna()
    no_matches_right = no_matches[["Unnamed: 0",'C1-B','C2-B','A1-B','A2-B',"Relation-B","Nuc-B","filename"]].dropna()
    add_no_matches1 = pd.merge(no_matches_left, no_matches_right, on="filename")

    additional_no_matches = rename_columns(add_no_matches1)

    rel_mismatch2_2, nuc_switched2_2, left_right_new_2, scopes_2, _ = more_tace_matches(additional_no_matches, tace_match=False, no_matches=no_matches)

    # Put everything together
    rel_mismatch = pd.concat([rel_mismatch, rel_mismatch2_2])
    nuc_switched = pd.concat([nuc_switched, nuc_switched2_2])
    left_right = pd.concat([left_right_new, left_right_new_2])
    scopes = pd.concat([scopes,scopes_2])
    all_other_categories = pd.concat([nrca_match, rel_mismatch, nuc_switched, left_right, scopes])
    no_matches = all_df.drop(all_other_categories.index)

    rel_mismatch["Category"] = "Interchangeable relations"
    nuc_switched["Category"] = "Nuclearity switched"
    nrca_match["Category"] = "Perfect match"
    left_right["Category"] = "EDU attached left or right"
    scopes["Category"] = "Different scope"
    no_matches["Category"] = "No match"

    all_with_categories = pd.concat([rel_mismatch, nuc_switched, nrca_match, left_right, scopes, no_matches])
    all_with_categories = all_with_categories[['ID','Category','Subcategory','CS-A','Relation-A','Nuc-A','C1-A','C2-A','CN-A','A1-A','A2-A','Text1a','Text1b','CS-B','Relation-B','Nuc-B','C1-B','C2-B','CN-B','A1-B','A2-B','Text2a','Text2b','Matching','N','R','C','A','Agreement','Disagreement','filename','span_length_1_all','span_length_2_all']]

    # Save to file
    all_with_categories.to_csv(output_file + ".csv")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-tace_path', type=str, default="../tace_output/", help='path to folder with the Tace Comparison files')
    parser.add_argument('-corpus_path', type=str, default="../data/1/", help='path to folder with the RST files')
    parser.add_argument('-output_file', type=str, default="output_file", help='name for the output CSV file')
    args = parser.parse_args()

    tace_df = create_tace_df(args.tace_path) 
    tace_df_with_text = add_edu_text(tace_df, args.corpus_path) 
    create_categories_pipeline(tace_df_with_text, args.output_file)    
