import pandas as pd
import random
from word2vec_split_data_to_categories import split_data

sickness_count_pre_ob = 0
non_sickness_count_pre_ob = 0
total_pre_ob = 0

training_data_pre_ob = []
other_data_pre_ob = []

training_data_post_ob = []
other_data_post_ob = []

sickness_count_post_ob = 0
non_sickness_count_post_ob = 0
total_post_ob = 0

number_of_samples = 500

def select_training_data(filename, training_data, other_data):
    count = 0
    total = 0
    with open(filename) as csvfile:
        reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["ID", "Created_at", "Location", "text", "label"])
        probability = number_of_samples / len(reader)
        print(probability)
        for index, row in reader.iterrows():
            print(total, end='\r')
            #print(line)
            # assume there's one document per line, tokens separated by whitespace
            # select for training
            total += 1
            prob = random.random()
            if prob <= probability:
                training_data.append(row)
                count += 1
            else:
                asd = row[["ID", "Created_at", "Location", "text"]]
                other_data.append(asd)
    return count

split_data()

print("Selecting training samples")

sickness_count_pre_ob += select_training_data("Binary_classification/sick_pre_ob.csv", training_data_pre_ob, other_data_pre_ob)
non_sickness_count_pre_ob += select_training_data("Binary_classification/other_pre_ob.csv", training_data_pre_ob, other_data_pre_ob)

print(type(training_data_pre_ob))

sickness_count_post_ob += select_training_data("Binary_classification/sick_post_ob.csv", training_data_post_ob, other_data_post_ob)
non_sickness_count_post_ob += select_training_data("Binary_classification/other_post_ob.csv", training_data_post_ob, other_data_post_ob)


pd.DataFrame(training_data_pre_ob).to_csv("Binary_classification/Training_data/training_data_pre_ob.csv")
pd.DataFrame(other_data_pre_ob).to_csv("Binary_classification/Training_data/other_data_pre_ob.csv")

pd.DataFrame(training_data_post_ob).to_csv("Binary_classification/Training_data/training_data_post_ob.csv")
pd.DataFrame(other_data_post_ob).to_csv("Binary_classification/Training_data/other_data_post_ob.csv")

        
print(f"Number of sickness related messages, pre-outbreak: {sickness_count_pre_ob}")
print(f"Number of non-sickness related messages, pre-outbreak: {non_sickness_count_pre_ob}")

print(f"Number of sickness related messages, post-outbreak: {sickness_count_post_ob}")
print(f"Number of non-sickness related messages, post-outbreak: {non_sickness_count_post_ob}")