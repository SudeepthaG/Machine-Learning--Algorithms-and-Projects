import csv
import re
import pandas as pd
import numpy as np
eps=np.finfo(float).eps   #epsilon value for use in log calculations
import pprint

def main():
    f=open("input.csv","r")
    fieldnames=f.readline()
    fieldnames=re.split(', |\n',fieldnames)
    fieldnames.pop()
    f.close()

    # reading data fom input csv file
    with open('input.csv') as csvfile:
        inputdata = list(csv.DictReader(csvfile,skipinitialspace=True,lineterminator=";"))

    #     writing cleaned data to csv file
    with open('cleaned_csv_file.csv', 'w') as csv_cleanfile:
        writer = csv.DictWriter(csv_cleanfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in inputdata:
            clean_first=row[fieldnames[0]].split(" ")[1]
            clean_last=row[fieldnames[-1]][:-1]
            row.update([(fieldnames[0],clean_first),(fieldnames[-1],clean_last)])
            writer.writerow(row)
    csv_cleanfile.close()

    #reading cleaned data and finding labels of attributes and writing to dataframe
    with open('cleaned_csv_file.csv') as csv_cleanfile:
        inputdata = list(csv.DictReader(csv_cleanfile))
    clean_fieldnames=list(fieldnames)
    clean_fieldnames[0]=fieldnames[0].split('(')[1]
    clean_fieldnames[-1] = fieldnames[-1].split(')')[0]
    df = pd.DataFrame(data=inputdata)
    df=df.rename(columns = {fieldnames[0]:clean_fieldnames[0]})
    df=df.rename(columns = {fieldnames[-1]:clean_fieldnames[-1]})
    print df
    target=clean_fieldnames[-1]
    df = df.reindex(columns=clean_fieldnames)
    dtree=buildTree(df,target)
    final_tree=pprint.pformat(dtree)
    print final_tree
    predict(dtree)

#   buildTree calls the necessary functions and build the decision tree from the cleaned input data
def buildTree(df,target,dtree=None):
    E=find_whole_entropy(target,df)
    attribute_entropy = {}
    for x in df.keys()[:-1]:
        attribute_entropy.update( {x:calculate_attribute_entropy(df,x,target)})
    max_info_gain_attribute=find_max_info_gain(E,df.keys()[:-1],attribute_entropy)
    labels=df[max_info_gain_attribute].unique()
    if dtree is None:
        dtree={}
        dtree[max_info_gain_attribute]={}
    else:
        print 'Tree got a new node:'+target

    for label in labels:
        new_df=df_after_splitting(df,max_info_gain_attribute,label)
        if new_df.equals(df):
            dtree[max_info_gain_attribute][label]='cant be determined due to noisy data'
            return dtree
        new_labels,count=np.unique(new_df[target],return_counts=True)
        if len(count)==1:
            dtree[max_info_gain_attribute][label]=new_labels[0]
        else:
            new_target = df.keys()[-1]
            dtree[max_info_gain_attribute][label] = buildTree(new_df,new_target)

    return dtree



# calculating entropy of whole dataset
def find_whole_entropy(target,df):
    E=0
    for x in df[target].value_counts():
        E=E+(float(x)/len(df[target])*np.log2(float(x)/len(df[target])))
    E=E*-1
    return E

#find_max_info_gain returns the attribute having the max information gain
def find_max_info_gain(E,clean_fieldnames,attribute_entropy):
    info_gain={}
    for x in clean_fieldnames:
        info_gain.update({x:E-attribute_entropy.get(x)})
    max_info_att=clean_fieldnames[0]
    for x in clean_fieldnames:
        if info_gain.get(x)>info_gain.get(max_info_att):
            max_info_att=x
    return max(info_gain,key=info_gain.get)   #returns attribute with highest information gain. If 2 or more attributes have the same information gain,
                                              # it returns the last occuring attribute with highest information gain. For eg, Music and VIP have same info_gain=1, it returns VIP


# calculate_attribute_entropy calculates and returns the entropy of each given attribute
def calculate_attribute_entropy(df,attribute,target):
    target_labels = df[target].unique()
    attribute_labels=df[attribute].unique()
    attribute_entropy=0
    for x in attribute_labels:
        e=0
        for y in target_labels:
            p=float(len(df[attribute][df[attribute]==x][df[target]==y]))/(len(df[attribute][df[attribute]==x])+eps)
            e+=-p*np.log2(p+eps)
        attribute_entropy+=(-float(len(df[attribute][df[attribute]==x])+eps)/len(df))*e
    return abs(attribute_entropy)


# df_after_splitting creates and returns a new dataframe after splitting based on attribute which has highest entropy
def df_after_splitting(df,attribute,label):
    new_df=df.loc[df[attribute]==label].reset_index(drop=True)
    return new_df

# predict takes the decision tree as input and predicts the label for predict_data
def predict(dtree):
    predict_data = {'Occupied': 'Moderate', 'Price': 'Cheap', 'Music': 'Loud', 'Location': 'City-Center', 'VIP': 'No', 'Favorite Beer': 'No'}
    print 'Current dtree:'
    final_tree=pprint.pformat(dtree)
    print final_tree
    print "Given prediction data:"
    print predict_data
    for attribute in dtree.keys():
        print '\nAttribute from decision tree:'+attribute
        value= predict_data.get(attribute)
        print 'Value from prediction data:'+value
        dtree=dtree.get(attribute).get(value)
        final_tree = pprint.pformat(dtree)
        if type(dtree)==dict:
            print '\nGOING ONE LEVEL DEEPER\n'
            prediction=predict(dtree)
        elif type(dtree)==type('Yes'):
            print '\nGOING ONE LEVEL DEEPER\n'
            print value+":"+dtree
            print '\nFINAL SOLUTION FOUND:'
            prediction=dtree
            print "Prediction:"+prediction
            break
        else:
            prediction='cant be determined due to lack of data'
            print prediction
            break
    return prediction

if __name__=="__main__":
    main()



