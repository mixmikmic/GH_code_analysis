import numpy as np
import pandas as pd
import json

founders = pd.read_csv("data/cleaned_input_data.csv", encoding="ISO-8859-1")
founders.fillna("")
founders.head()

output = pd.read_csv("data/cleaned_output_data.csv", encoding="ISO-8859-1")
output.head()

# Load in weights for each feature (generated using RandomForestRegressor)
weight_df = pd.read_csv("data/weights.csv", encoding="ISO-8859-1")
weight_df.head()

weights = weight_df["Importance"]

class FounderSimilarityCalculator:
    def __init__(self, data, weights):
        """
        Input:
            data: pandas dataframe of feature values (assumes founder name is first column, company name is second column)
            weights: array of weights for each feature to be used in determining "distance" between founders
        """
        self.founders = data.iloc[:,0] # the first column is assumed to be founder name
        self.companies = data.iloc[:,1] # the second column is assumed to be company name
        self.crunchbase = data.iloc[:,18] # the 19th column is assumed to be the founder's CrunchBase link
        self.linkedin = data.iloc[:,19] # the 20th column is assumed to be the founder's LinkedIn link
        
        self.features = data.iloc[:,2:18]
        self.weights = weights
        self.sum_of_weights = np.sum(weights) # compute the sum of the weights as a normalizing factor for similarity
        
        assert self.features.shape[1] == len(self.weights) # ensure that the # of weights corresponds to # of features
    
    def _weightedContributionToSimilarity(self, feature1, feature2, weight):
        """
        Computes the weighted contribution of the current feature to the similarity measure.
        
        If the two values are the same, then we add 1*weight to the similarity score.
        
        For features whose values are continuous, the probability of 2 values being equal is very small, so we don't want
        to penalize them for being different. We can use a smoother penalty based on how different they are. So, if the 
        two values are different,
            1. Compute the absolute difference between the two values
            2. Add 1/difference * weight to the similarity score
        
        Input:
            feature1: feature of founder
            feature2: corresponding feature of different founder
            weight: weight on the feature
        """
        if isinstance(feature1, str) and isinstance(feature2, str):
            if feature1 == feature2:
                return weight
            else:
                return 0
        elif isinstance(feature1, float) and isinstance(feature2, float):
            if not (np.isnan(feature1) or np.isnan(feature2)):
                diff = np.abs(feature1 - feature2)
                if diff <= 1:
                    return weight
                else:
                    return 1/diff * weight
            else:
                return 0
        return 0
    
    def _computeWeightedSimilarity(self, founder1_data, founder2_index):
        """
        Computes the weighted similarity between 2 founders.
        
        Input:
            founder1_data: information (feature values) of founder 1
            founder2_index: integer index of founder 2
        """
        features1 = founder1_data
        features2 = self.features.iloc[founder2_index,:]
        
        similarity = 0
        for i in range(features1.shape[0]):
            similarity += self._weightedContributionToSimilarity(features1[i], features2[i], self.weights[i])
        
        return similarity/self.sum_of_weights
    
    def findKClosestFounders(self, k, founder_data):
        """
        Finds the k closest founders in terms of similarity to the founder corresponding to founder_index.
        
        Input:
            k: # of most similar founders
            founder_data: information (feature values) of founder of whom we wish to find similar founders
        """
        assert k < self.founders.shape[0]
        
        similarity = np.zeros(self.founders.shape[0])
        
        for i in range(self.founders.shape[0]):
            similarity[i] = self._computeWeightedSimilarity(founder_data, i)
            
        min_indices = similarity.argsort()[::-1][:k]
        
        closest_founders = []
        closest_companies = []
        similarity_score = []
        linkedin_links = []
        
        for i in min_indices:
            closest_founders.append(self.founders.iloc[i])
            closest_companies.append(self.companies.iloc[i])
            similarity_score.append(similarity[i])
            linkedin_links.append(self.linkedin.iloc[i])
        
        return closest_founders, closest_companies, similarity_score, linkedin_links

fsc = FounderSimilarityCalculator(founders, weights)

# Find the 5 closest founders
founder_data = {"Full Name": "Stephen Torres",
               "Primary Company": "PV Solar Report",
               "Previous startups?": 1,
               "Consulting before start-up": 0,
               "Standardized University": "University of California Berkeley",
               "Standardized Major": "Business",
               "Degree Type": "BA",
               "Standardized Graduate Institution": "Cornell University",
               "Standradized Graduate Studies": "Business",
               "Graduate Diploma": "MBA",
               "Ever served as TA/Teacher/Professor/Mentor?": 1,
               "Years of Employment": 9,
               "Worked as product manager/director/head/VP?": 0,
               "Worked at Google?": 0,
               "Worked at Microsoft": 0,
               "Worked in Sales?": 1,
               "Stanford or Berkeley": 1,
               "Ivy League": 1,
               "Crunchbase": "",
               "LinkedIn": "https://www.linkedin.com/in/stephendtorres/"}

k = 5
closest_founders, closest_companies, similarity_score, linkedin = fsc.findKClosestFounders(k, 
                                                                            np.array(list(founder_data.values())[2:18]))

closest = list(zip(closest_founders, closest_companies, similarity_score, linkedin))

data = {}
data["name"] = founder_data["Full Name"]
data["company"] = founder_data["Primary Company"]
data["size"] = 3000
data["link"] = founder_data["LinkedIn"]

data["children"] = []
for tup in closest:
    name = tup[0]
    company = tup[1]
    similarity = round(tup[2], 4)
    linkedin = tup[3] if isinstance(tup[3], str) else ""
    size = tup[2]*(10**4)
    
    new_dict = {"name": name,
               "company": company,
               "similar": similarity,
               "link": linkedin,
               "size": size,}
    data["children"].append(new_dict)

with open('front-end-UI/data.json', 'w') as outfile:  
    json.dump(data, outfile)

with open('front-end-UI/data.json') as infile:
    data = json.load(infile)
    
    for key, val in data.items():
        if key != "children":
            print(key, ":", val)
        else:
            children = data["children"]
            print("\n***************Most similar founders:***************\n")
            for founder_info in children:
                for key, val in founder_info.items():
                    print(key, ":", val)
                print()

