import pandas as pd
import random
import re
import csv
from word2vec_helpers import get_pre_ob_regex, get_word_list


regex = get_pre_ob_regex()
word_list = get_word_list()
total_count = 0

sickness_count_pre = 0
non_sickness_count_pre = 0

sickness_count_post = 0
non_sickness_count_post = 0

training_data = []
all_data = []

sick_pre_ob = []
other_pre_ob = []
sick_post_ob = []
other_post_ob = []

blacklist = ["fried chicken flu", 'weather', "influenc", "influential"]


with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["ID", "Created_at", "Location", "text"])

    for index, row in reader.iterrows():
        print(total_count, end='\r')
        text = row["text"].lower()
        if (re.match(regex, row["Created_at"])):
            total_count += 1
            if any(substring in text for substring in word_list):
                if any(substring in text for substring in blacklist):
                    non_sickness_count_pre += 1
                    row["label"] = 0
                    other_pre_ob.append(row)
                else:
                    sickness_count_pre += 1
                    row["label"] = 1
                    sick_pre_ob.append(row)
            else:
                non_sickness_count_pre += 1
                row["label"] = 0
                other_pre_ob.append(row)
        else:
            total_count += 1
            print(total_count, end='\r')
            if any(substring in text for substring in word_list):
                if any(substring in text for substring in blacklist):
                    non_sickness_count_post += 1
                    row["label"] = 0
                    other_post_ob.append(row)
                else:
                    sickness_count_post += 1
                    row["label"] = 1
                    sick_post_ob.append(row)
            else:
                non_sickness_count_post += 1
                row["label"] = 0
                other_post_ob.append(row)

pd.DataFrame(sick_pre_ob).to_csv("Binary_classification/sick_pre_ob.csv")
pd.DataFrame(other_pre_ob).to_csv("Binary_classification/other_pre_ob.csv")

pd.DataFrame(sick_post_ob).to_csv("Binary_classification/sick_post_ob.csv")
pd.DataFrame(other_post_ob).to_csv("Binary_classification/other_post_ob.csv")

print(f"Number of sickness messages pre-outbreak: {sickness_count_pre}")
print(f"Number of other messages pre-outbreak: {non_sickness_count_pre}")

print(f"Number of sickness messages post-outbreak: {sickness_count_post}")
print(f"Number of other messages post-outbreak: {non_sickness_count_post}")

'''
with open("Binary_classification/sick_pre_ob.csv", 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    # writing headers (field names)
    writer.writeheader()
    # writing data rows
    writer.writerows(sick_pre_ob)
    print(f"Number of sickness messages: {len(sick_pre_ob)}")

with open("Binary_classification/non_pre_ob.csv", 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames = fields)

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(other_pre_ob)
    print(f"Number of other messages: {len(other_pre_ob)}")
'''

'''with open("Binary_classification/pre_ob.csv", "w") as file:
    for line in lines:
        file.write(f"{line}\n")
'''

'''
with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["ID", "Created_at", "Location", "text"])
    for index, row in reader.iterrows():
        #print(line)
        # assume there's one document per line, tokens separated by whitespace

        # select for training
        if (re.match(regex, row["Created_at"])):
            prob = random.randint(0, 10000)
            if prob <= 10:
                training_data.append(row)
                total_count += 1
                if any(substring in row["text"] for substring in word_list):
                    sickness_count += 1
                    row["label"] = 1
                else:
                    non_sickness_count += 1
                    row["label"] = 0
            else:
                all_data.append(row)
        else:
            all_data.append(row)

print(f"Size of training data: {total_count}")
print(f"Number of sickness related messages: {sickness_count}")
print(f"Number of non-sickness related messages: {non_sickness_count}")
'''