from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import numpy as np
import pickle
from joblib import dump, load
from data_preprocess import *


def model_fit_predict(clf, train_data, train_label, test_data, test_label):
    clf.fit(train_data, train_label)
    print("training accuracy: ", clf.score(train_data, train_label))
    print("testing accuracy: ", clf.score(test_data, test_label))
    test_predict = clf.predict(test_data)
    test_predict_prob = clf.predict_proba(test_data)
    print("testing auc score: ", roc_auc_score(test_label, test_predict_prob[:, 1]))
    print(classification_report(test_label, test_predict))
    return clf


def data_vect_fit(df, column, min_gram=1, max_grams=2, **kwargs):
    transform_collection = {}
    tf_idf_model = TfidfVectorizer(min_df=0.03, max_df=0.97, max_features=1000, ngram_range=(min_gram, max_grams))
    tf_idf_output = tf_idf_model.fit_transform(df[column])
    transform_collection['tf_idf'] = tf_idf_model
    result = tf_idf_output.toarray()
    if kwargs.get('sentiment', None):
        sentiment_output = df[['polarity', 'subjectivity']].values
        result = np.concatenate([result, sentiment_output], axis=1)
        transform_collection['sentiment'] = 'blob'
    if kwargs.get('topic_vect', None):
        vectorizer = CountVectorizer()
        count_vec = vectorizer.fit_transform(df[column])
        topic_num = kwargs.get('topic_num', 20)
        lda_model = LatentDirichletAllocation(n_components=topic_num, learning_method='online', n_jobs=-1)
        lda_output = lda_model.fit_transform(count_vec)
        transform_collection['topic_model'] = (vectorizer, lda_model)
        result = np.concatenate([result, lda_output], axis=1)
    return result, transform_collection


def data_vect_transform(df, column, transform_collection: dict):
    result = []
    for key, value in transform_collection.items():
        if key == 'tf_idf':
            result.append(value.transform(df[column]).toarray())
        elif key == 'sentiment':
            df['polarity'] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
            df['subjectivity'] = df[column].apply(lambda x: TextBlob(x).sentiment.subjectivity)
            result.append(df[['polarity', 'subjectivity']].values)
        else:
            vectorizer, lda_model = value
            count_vec = vectorizer.transform(df[column])
            lda_output = lda_model.transform(count_vec)
            result.append(lda_output)
    result = np.concatenate(result, axis=1)
    return result


def save_transform(transform_collection, filename):
    with open(filename, 'wb') as f:
        pickle.dump(transform_collection, f)


def load_transform(filename):
    with open(filename, 'rb') as f:
        transform = pickle.load(f)
    return transform


results = {0: 'Down', 1: "Up"}

#
# class InferenceModel:
#     def __init__(self):
#         self.transform = load_transform('transform.m')
#         self.model = load('rf.m')
#
#     def inference(self, test_data):
#         # test_df = pd.DataFrame([test_data], columns=['summary'])
#         # test_df = data_preprocess(test_df, 'summary', remove_stopwords=True, sentiment=True, stem_lemma=False)
#         # test_data = data_vect_transform(test_df, CLEAN_COL, self.transform)
#         html, test_pred, test_pred_prob = inference_show(test_data, self.model, self.transform)
#         prediction = {}
#         for idx, prob in enumerate(test_pred_prob):
#             prediction[results[idx]] = prob
#         return html, prediction.items()
#
#
# if __name__ == '__main__':
#     model = InferenceModel()
#     test_data = 'georgia downs two russian warplanes countries move brink war breaking musharraf impeached russia ' \
#                 'today columns troops roll south ossetia footage fighting youtube russian tanks moving towards ' \
#                 'capital south ossetia reportedly completely destroyed georgian artillery fire afghan children raped ' \
#                 'impunity u n official says sick three year old raped nothing russian tanks entered south ossetia ' \
#                 'whilst georgia shoots two russian jets breaking georgia invades south ossetia russia warned would ' \
#                 'intervene sos side enemy combatent trials nothing sham salim haman sentenced years kept longer ' \
#                 'anyway feel like georgian troops retreat osettain capital presumably leaving several hundred people ' \
#                 'killed video u prep georgia war russia rice gives green light israel attack iran says u veto israeli ' \
#                 'military ops announcing class action lawsuit behalf american public fbi sorussia georgia war nyts ' \
#                 'top story opening ceremonies olympics fucking disgrace yet proof decline journalism china tells bush ' \
#                 'stay countries affairs world war iii start today georgia invades south ossetia russia gets involved ' \
#                 'nato absorb georgia unleash full scale war alqaeda faces islamist backlash condoleezza rice us would ' \
#                 'act prevent israeli strike iran israeli defense minister ehud barak israel prepared uncompromising ' \
#                 'victory case military hostilities busy day european union approved new sanctions iran protest ' \
#                 'nuclear programme georgia withdraw soldiers iraq help fight russian forces georgias breakaway region ' \
#                 'south ossetia pentagon thinks attacking iran bad idea us news amp world report caucasus crisis ' \
#                 'georgia invades south ossetia indian shoe manufactory series like work visitors suffering mental ' \
#                 'illnesses banned olympics help mexicos kidnapping surge'
#     _, prediction = model.inference(test_data)
#     print(prediction)
