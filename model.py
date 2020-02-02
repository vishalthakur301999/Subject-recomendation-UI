import pickle
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv("marks.csv",index_col=0)
def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row
ratings_std = ratings.apply(standardize)
item_similarity = cosine_similarity(ratings_std.T)
item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns)
def get_similar_subjects(subject_name, user_rating):
    similar_score = item_similarity_df[subject_name] * user_rating
    # print(type(similar_ratings))
    return similar_score
pickle.dump(get_similar_subjects,open('model.pkl','wb'))