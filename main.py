import cudf
import xgboost
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import numpy as np
import os
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from gensim.models import Word2Vec
import shap

# Load environment variables
load_dotenv()
# Create a connection to the database
connection = create_engine(os.getenv("DATABASE_URL"))
# Query the database
query = text("SELECT * FROM auctions WHERE bin = TRUE")
res = connection.connect().execute(query)
# Convert the result to a pandas dataframe
df = pd.DataFrame(res, columns=res.keys())
# Create Data Preperation Pipeline
# Drop auctioneer, profile_id, coop, item_uuid, item_lore, unbreakable, claimed, last_updated,
df = df.drop(
    columns=[
        "auctioneer",
        "profile_id",
        "coop",
        "item_uuid",
        "unbreakable",
        "claimed",
        "last_updated",
        "bin",
    ]
)
# Fill in missing reforges with "None"
df["reforge"] = df["reforge"].fillna("None")
# Go through all enchantments, if they are a list of strings, add each to the corpus list otherwise just add the singular string to the corpus list
# Rather than using item_id as a categorial data, we use median and lowest price
df["lowest_bin"] = df.groupby("item_id")["price"].transform("min")
df["median_bin"] = df.groupby("item_id")["price"].transform("median")
# Drop item_id
df = df.drop(columns=["item_id"])
# preprocess pet held item
df["pet_held_item"] = df["pet_held_item"].fillna("No Pet Held Item")

# turn pet held item into a onehot encoded column
one_hot_encoder = OneHotEncoder(sparse_output=False)
pet_held_item_encoded = one_hot_encoder.fit_transform(df[["pet_held_item"]].to_numpy())
df = df.join(
    pd.DataFrame(
        pet_held_item_encoded, columns=one_hot_encoder.categories_[0], index=df.index
    )
)
df = df.drop("pet_held_item", axis=1)
# preprocess tier.
label_encoder = LabelEncoder()
df["tier"] = label_encoder.fit_transform(df["tier"].to_numpy())


# preprocess category.
one_hot_encoder = OneHotEncoder(sparse_output=False)
category_reshaped = df["category"].to_numpy().reshape(-1, 1)

# Fit and transform 'category'
category_encoded = one_hot_encoder.fit_transform(category_reshaped)

# Create a DataFrame from the encoded data
# Flatten the categories array to get the correct column names
category_encoded_df = pd.DataFrame(
    category_encoded, columns=one_hot_encoder.categories_[0], index=df.index
)

# Drop the original 'category' column
df.drop("category", axis=1, inplace=True)

# Join the new DataFrame with the original one
df = df.join(category_encoded_df)

# preprocess pet type.
one_hot_encoder = OneHotEncoder(sparse_output=False)
df["pet_type"] = df["pet_type"].fillna("NA")
pet_type_encoded = one_hot_encoder.fit_transform(df[["pet_type"]].to_numpy())
df.drop("pet_type", axis=1, inplace=True)
# Create a DataFrame from the encoded data
pet_type_encoded_df = pd.DataFrame(
    pet_type_encoded, columns=one_hot_encoder.categories_[0], index=df.index
)
# Join the new DataFrame with the original one
df = df.join(pet_type_encoded_df)

# preprocess reforge. (one hot)
one_hot_encoder = OneHotEncoder(sparse_output=False)
reforge_encoded = one_hot_encoder.fit_transform(df[["reforge"]].to_numpy())
df.drop("reforge", axis=1, inplace=True)
# Create a DataFrame from the encoded data
# Use [0] to access the list of categories for the 'reforge' column
reforge_encoded_df = pd.DataFrame(
    reforge_encoded, columns=one_hot_encoder.categories_[0], index=df.index
)
# Join the new DataFrame with the original one
df = df.join(reforge_encoded_df)

# preprocess enchantments. (word2vec)
# Convert the enchantments column to a list of lists
df["enchantments"] = df["enchantments"].fillna("No Enchantments")
corpus = df["enchantments"].tolist()
# Create the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
# Create a DataFrame from the Word2Vec model


def average_vectors(item):
    vectors = [model.wv[enchantment] for enchantment in item if enchantment in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


df["enchantment_vector"] = df["enchantments"].apply(average_vectors)
# Join the new DataFrame with the original one_hot_encoder
vectors_df = pd.DataFrame(df["enchantment_vector"].tolist(), index=df.index).add_suffix(
    "_enchs_vector"
)
# You can now concatenate this with your original DataFrame to have each vector dimension as a separate column
df = pd.concat([df, vectors_df], axis=1)
# Drop the original 'enchantments' column
df.drop("enchantments", axis=1, inplace=True)
df.drop("enchantment_vector", axis=1, inplace=True)
print("Finished Preprocessing Enchantments")
print("Starting Preprocessing Item Name")
# preprocess item_name. (word2vec)
corvus = df["item_name"].tolist()
# Create the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
# Create a DataFrame from the Word2Vec model
df["item_name_vector"] = df["item_name"].apply(average_vectors)
vectors_df = pd.DataFrame(df["item_name_vector"].tolist(), index=df.index).add_suffix(
    "_item_name_vector"
)
# You can now concatenate this with your original DataFrame to have each vector dimension as a separate column
df = pd.concat([df, vectors_df], axis=1)
# Drop the original 'item_name' column
df.drop("item_name", axis=1, inplace=True)
df.drop("item_name_vector", axis=1, inplace=True)
print("Finished Preprocessing Item Name")
print("Starting Preprocessing item lore")
# preprocess item_lore. (word2vec)
corpus = df["item_lore"].tolist()
# Create the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
# Create a DataFrame from the Word2Vec model
df["item_lore_vector"] = df["item_lore"].apply(average_vectors)
vectors_df = pd.DataFrame(df["item_lore_vector"].tolist(), index=df.index).add_suffix(
    "_item_lore_vector"
)
# You can now concatenate this with your original DataFrame to have each vector dimension as a separate column
df = pd.concat([df, vectors_df], axis=1)
# Drop the original 'item_lore' column
df.drop("item_lore", axis=1, inplace=True)
df.drop("item_lore_vector", axis=1, inplace=True)
print("Finished Preprocessing Item Lore")
print("Starting Preprocessing Slotted Gems")
# preprocess slotted_gems. (word2vec)
df["slotted_gems"] = df["slotted_gems"].fillna("No Slotted Gems")
temp = df["slotted_gems"]
corpus = temp.tolist()
# Create the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
# Create a DataFrame from the Word2Vec model
df["slotted_gems_vector"] = df["slotted_gems"].apply(average_vectors)
vectors_df = pd.DataFrame(
    df["slotted_gems_vector"].tolist(), index=df.index
).add_suffix("_slotted_gems_vector")
# You can now concatenate this with your original DataFrame to have each vector dimension as a separate column
df = pd.concat([df, vectors_df], axis=1)
# Drop the original 'slotted_gems' column
df.drop("slotted_gems", axis=1, inplace=True)
df.drop("slotted_gems_vector", axis=1, inplace=True)
print("Finished Preprocessing Slotted Gems")
print("Starting Preprocessing unlocked_gem_slots")
# preprocess unlocked_gem_slots. (word2vec)
print(df["unlocked_gem_slots"])
df["unlocked_gem_slots"] = df["unlocked_gem_slots"].fillna("No Unlocked Gem Slots")
corpus = df["unlocked_gem_slots"].tolist()
# Create the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
# Create a DataFrame from the Word2Vec model
df["unlocked_gem_slots_vector"] = df["unlocked_gem_slots"].apply(average_vectors)
vectors_df = pd.DataFrame(
    df["unlocked_gem_slots_vector"].tolist(), index=df.index
).add_suffix("_unlocked_gem_slots_vector")
# You can now concatenate this with your original DataFrame to have each vector dimension as a separate column
df = pd.concat([df, vectors_df], axis=1)
# Drop the original 'unlocked_gem_slots' column
df.drop("unlocked_gem_slots", axis=1, inplace=True)
df.drop("unlocked_gem_slots_vector", axis=1, inplace=True)
print("Finished Preprocessing Unlocked Gem Slots")
print("Starting Preprocessing Runes")
# preprocess runes. (word2vec)
df["runes"] = df["runes"].fillna("No Runes")
corpus = df["runes"].tolist()
# Create the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
# Create a DataFrame from the Word2Vec model
df["runes_vector"] = df["runes"].apply(average_vectors)
vectors_df = pd.DataFrame(df["runes_vector"].tolist(), index=df.index).add_suffix(
    "_runes_vector"
)
# You can now concatenate this with your original DataFrame to have each vector dimension as a separate column
df = pd.concat([df, vectors_df], axis=1)
# Drop the original 'runes' column
df.drop("runes", axis=1, inplace=True)
df.drop("runes_vector", axis=1, inplace=True)
print("Finished Preprocessing Runes")
# Convert start and end time into day, month, year, hour and minute
df["start_time"] = pd.to_datetime(df["start_time"])
df["end_time"] = pd.to_datetime(df["end_time"])
df["start_day"] = df["start_time"].dt.day
df["start_month"] = df["start_time"].dt.month
df["start_year"] = df["start_time"].dt.year
df["start_hour"] = df["start_time"].dt.hour
df["start_minute"] = df["start_time"].dt.minute
df["end_day"] = df["end_time"].dt.day
df["end_month"] = df["end_time"].dt.month
df["end_year"] = df["end_time"].dt.year
df["end_hour"] = df["end_time"].dt.hour
df["end_minute"] = df["end_time"].dt.minute
df = df.drop(columns=["start_time", "end_time"])
print("Finished Preprocessing Time")
# Fill all missing values with -1 to represent not existing
df = df.fillna(-1)
# Change TRUE and FALSE to 1 and 0
df = df.replace({True: 1, False: 0})
print("Converting to cuDF and Training Model")
print(df.columns)
# Check for any categorial data left behind by accident
print(df.select_dtypes(include=["object"]).columns)
# Convert all types into int
df = df.drop(["uuid"], axis=1)
# Remove indexes
df = df.reset_index(drop=True)
df.to_csv("data.csv", index=False)
cudf_df = cudf.DataFrame.from_pandas(df)
y = cudf_df["price"]
X = cudf_df.drop(["price"], axis=1)

# Create the model.
model = xgboost.XGBRegressor(
    tree_method="hist",
    device="cuda",
    n_estimators=800,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=10,
)
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train the model
model.fit(X_train, y_train)
# Cross Val Score using scikit
scores = cross_val_score(model, X, y.to_numpy(), cv=5, scoring="neg_median_absolute_error")
# Print the mean squared error
print(scores)
