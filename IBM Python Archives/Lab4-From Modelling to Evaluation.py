#Import Libraries
import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
import re
import random
import aiohttp
import asyncio

async def download(url, filename):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(filename, "wb") as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    print(f"Downloaded {url} to {filename}")
                else:
                    print(f"Failed to download {url}. Status code: {response.status}")
    except aiohttp.ClientError as e:
        print(f"An error occurred: {e}")

async def main():
    path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv"
    filename = "recipes.csv"
    await download(path, filename)

# Run the asynchronous code
if __name__ == "__main__":
    asyncio.run(main())

recipes = pd.read_csv("recipes.csv")
print("Data read into dataframe!")

#Fix name of the column displaying the cuisine 
column_names = recipes.columns.values
print(column_names)
column_names[0] = "cuisine"
recipes.columns = column_names
print(recipes.columns)

#convert cuisine names to lower case 
recipes['cuisine'] = recipes['cuisine'].str.lower()

print(recipes.head())

#make the cuisine name consistent 
recipes.loc[recipes["cuisine"] == "austria", "cuisine"] = "austrian"
recipes.loc[recipes["cuisine"] == "belgium", "cuisine"] = "belgian"
recipes.loc[recipes["cuisine"] == "china", "cuisine"] = "chinese"
recipes.loc[recipes["cuisine"] == "canada", "cuisine"] = "canadian"
recipes.loc[recipes["cuisine"] == "netherlands", "cuisine"] = "dutch"
recipes.loc[recipes["cuisine"] == "france", "cuisine"] = "french"
recipes.loc[recipes["cuisine"] == "germany", "cuisine"] = "german"
recipes.loc[recipes["cuisine"] == "india", "cuisine"] = "indian"
recipes.loc[recipes["cuisine"] == "indonesia", "cuisine"] = "indonesian"
recipes.loc[recipes["cuisine"] == "iran", "cuisine"] = "iranian"
recipes.loc[recipes["cuisine"] == "italy", "cuisine"] = "italian"
recipes.loc[recipes["cuisine"] == "japan", "cuisine"] = "japanese"
recipes.loc[recipes["cuisine"] == "israel", "cuisine"] = "jewish"
recipes.loc[recipes["cuisine"] == "korea", "cuisine"] = "korean"
recipes.loc[recipes["cuisine"] == "lebanon", "cuisine"] = "lebanese"
recipes.loc[recipes["cuisine"] == "malaysia", "cuisine"] = "malaysian"
recipes.loc[recipes["cuisine"] == "mexico", "cuisine"] = "mexican"
recipes.loc[recipes["cuisine"] == "pakistan", "cuisine"] = "pakistani"
recipes.loc[recipes["cuisine"] == "philippines", "cuisine"] = "philippine"
recipes.loc[recipes["cuisine"] == "scandinavia", "cuisine"] = "scandinavian"
recipes.loc[recipes["cuisine"] == "spain", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "portugal", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "switzerland", "cuisine"] = "swiss"
recipes.loc[recipes["cuisine"] == "thailand", "cuisine"] = "thai"
recipes.loc[recipes["cuisine"] == "turkey", "cuisine"] = "turkish"
recipes.loc[recipes["cuisine"] == "vietnam", "cuisine"] = "vietnamese"
recipes.loc[recipes["cuisine"] == "uk-and-ireland", "cuisine"] = "uk-and-irish"
recipes.loc[recipes["cuisine"] == "irish", "cuisine"] = "uk-and-irish"

print(recipes.head())

#remove data for cuisine with less than 50 recipes: 
recipes_counts = recipes["cuisine"].value_counts()
print(recipes_counts)
cuisine_indices = recipes_counts > 50

#recipe_count.index.value - extracts the index (cuisines) of the recipes_counts series and converts it to a numpy array 
test = recipes_counts.index.values
print(test)
#This converts the boolean series cuisine_indices to a Numpy Array of boolean values 
test2 = np.array(cuisine_indices)
print(type(cuisine_indices))
#class pandas.core.series.series - pandas series containing boolean values True and False - alter: class numpy.ndarray - numpy array containing boolean values
#uses the boolean array to index the array of cuisines, returning only the cuisines that appear more than 50 times 
np.array(recipes_counts.index.values)[np.array(cuisine_indices)]
#list() converts the resulting array into a python list 
print(test2)



cuisine_to_keep = list(np.array(recipes_counts.index.values)[np.array(cuisine_indices)])
recipes = recipes.loc[recipes['cuisine'].isin(cuisine_to_keep)]

recipes = recipes.replace(to_replace="Yes", value = 1)
recipes = recipes.replace(to_replace="No", value = 0)

## DATA MODELLING ##

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

recipes.head()

#creating a decision tree for the recipes for just some of the asian and indian cuisines 
#select subset of cuisines 
asian_indian_recipes= recipes[recipes.cuisine.isin(['korean','japanese','chinese','thai','indian'])]
#or recipes[recipes['cuisine'].isin(['korean','japanese','chinese','thai','indian'])]
cuisines = asian_indian_recipes["cuisine"]
#selects the second column onwards
ingredients = asian_indian_recipes.iloc[:,1:]

bamboo_tree = tree.DecisionTreeClassifier(max_depth=3)
bamboo_tree.fit(ingredients,cuisines)

print('Decision tree model saved to bamboo_tree!')

#Let's plot the decison tree and examine what it looks like

plt.figure(figsize=(60,40)) #customize according to the size of the tree
_=tree.plot_tree(bamboo_tree,
                 feature_names=list(ingredients.columns.values),
                 class_names=np.unique(cuisines),filled=True,
                 node_ids=True,
                 impurity=False,
                 label='all',
                 fontsize=20, rounded=True)
plt.show()