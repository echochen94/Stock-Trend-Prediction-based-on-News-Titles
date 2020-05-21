from IPython.display import HTML
from IPython.display import display
from ipywidgets import widgets
from joblib import dump, load
import numpy as np
from data_preprocess import *
from model import *


def get_data_from_input():
    text = widgets.Text(
        value='',
        placeholder='Input the test data here!',
        description='Input:',
        disabled=False)
    display(text)

    def callback(wdgt):
        print(wdgt.value)

    text.on_submit(callback)
    return text



def test_data_preprocess(test_df):
    for column in test_df.columns[2:]:
        test_df[column] = test_df[column].str.lstrip('b')
    test_df['summary'] = test_df[test_df.columns[2:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    test_df = data_preprocess(test_df, 'summary', remove_stopwords=True, stem_lemma=False)
    return test_df


def inference_show(test_data, clf, transform):
    df = pd.DataFrame([test_data], columns=['summary'])
    df = data_preprocess(df, 'summary', remove_stopwords=True, sentiment=True, stem_lemma=False)
    sentiment_info = []
    sentiment_info.append(df['polarity'].iloc[0])
    sentiment_info.append(df['subjectivity'].iloc[0])
    tf_idf = transform['tf_idf']
    word_importance = {}
    for word, feature_importamce in zip(tf_idf.vocabulary_, clf.feature_importances_[:len(tf_idf.vocabulary_)]):
        word_importance[word] = feature_importamce

    data_vec = data_vect_transform(df, CLEAN_COL, transform)
    prediction = clf.predict(data_vec)[0]
    predict_prob = clf.predict_proba(data_vec)[0]

    analyzer = tf_idf.build_analyzer()
    tokenizer = tf_idf.build_tokenizer()
    heatmap = heatmap_generate(analyzer, tokenizer, test_data, word_importance)
    html = ""
    # html += "<span><h3>Based on the input news title, the model predicts as {}".format(results[prediction])
    # html += "<small><br>Confidence: {:.0f}%<br><br></small></h3></span>".format(abs(((predict_prob * 100))))
    for index, token in enumerate(tokenizer(test_data)):
        html += "<span style='background-color:rgba({},0,150,{})'>{} </span>".format(heatmap[index] * 255,
                                                                                     heatmap[index] - 0.3, token)
    HTML(html)
    return html, prediction, predict_prob, sentiment_info


def heatmap_generate(analyzer, tokenizer, data, word_importance):
    heatmap = np.zeros(len(tokenizer(data)))
    word_len = len(tokenizer(data))
    for idx, token in enumerate(analyzer(data)):
        if idx < word_len:
            if token in word_importance:
                heatmap[idx] += word_importance[token]
        else:
            if token in word_importance:
                heatmap[idx - word_len] += word_importance[token]
                heatmap[idx - word_len + 1] += word_importance[token]

    heatmap = heatmap / np.max(heatmap)
    return heatmap

class InferenceModel:
    def __init__(self):
        self.transform = load_transform('transform.m')
        self.model = load('rf.m')

    def inference(self, test_data):
        # test_df = pd.DataFrame([test_data], columns=['summary'])
        # test_df = data_preprocess(test_df, 'summary', remove_stopwords=True, sentiment=True, stem_lemma=False)
        # test_data = data_vect_transform(test_df, CLEAN_COL, self.transform)
        html, test_pred, test_pred_prob, sentiment = inference_show(test_data, self.model, self.transform)
        prediction = {}
        for idx, prob in enumerate(test_pred_prob):
            prediction[results[idx]] = prob
        return html, prediction.items(), [sentiment]
