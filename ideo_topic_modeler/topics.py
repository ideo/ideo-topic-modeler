# from pathlib import Path
from datetime import datetime

import pandas as pd
import altair as alt
import plotly.express as px
from umap import UMAP
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import plotly.express as px
from streamlit_plotly_events import plotly_events

from ideo_topic_modeler import Model


TODAY = datetime.now().strftime("%d_%m_%Y_%H%M%S")
TODAY_DATE = datetime.now().date().strftime("%d_%m_%Y")


class TopicModel(Model):

    def __init__(self, data, text_column, data_source, model_directory, language="english"):
        '''
        Initializes an instance of the TopicModel class.

        Parameters
        ----------
        data: pandas DataFrame instance
            The dataframe with a column to be used for n-gram analysis.
        text_column: str
            The name of the column to be used for topic analysis.
        data_source: str
            where the data are coming from
        '''
        super(TopicModel, self).__init__(data, text_column, data_source, language)

        # today = datetime.now().strftime("%d_%m_%Y_%H%M%S")

        self.model_directory = model_directory
        # self.data_filename = self.model_directory/ f"data_{today}.json"        

        #FIXME what's the best way to do this?
        # if self.data_source == 'reddit':
        self.pre_trained_model = "paraphrase-mpnet-base-v2"


    def run(self):
        """This function compute topics and embeddings.
        """
        sentence_model = SentenceTransformer(self.pre_trained_model)
        self.embeddings = sentence_model.encode(self._get_corpus(), show_progress_bar=True)
        self.topic_model = BERTopic()


    def save_model(self, my_timestamp = TODAY):
        """This function saves model and embeddings.
        """
        pd.DataFrame(self.embeddings).to_json(self.model_directory/ f"embeddings_{my_timestamp}.json", orient='records', lines=True)
        self.topic_model.save(self.model_directory/ f"model_{my_timestamp}")

        # #FIXME when is this ever called with save_data = True?
        # if save_data:
        #     self.save_data(my_timestamp)
        
    def save_data(self, my_timestamp):
        """Saves the data after any modifications (like clustering)."""
        self.data_filename = self.model_directory/ f"data_{my_timestamp}.json"
        self.data.to_json(self.data_filename, orient='records', lines=True)
    
    def enrich_data_and_save_them(self, my_timestamp = TODAY):
        """This function adds to the data the topics information and saves them into a json file.
        """

        topics, probs = self.topic_model.fit_transform(self._get_corpus(), self.embeddings)

        #enriching the data with the model info and tf_idf
        self.data.loc[:, 'topic'] = topics        
        self.data.loc[:, 'probability'] = probs        
        
        topic_name_map = dict(zip(self.topic_model.get_topic_info()['Topic'].tolist(), 
                                  self.topic_model.get_topic_info()['Name'].tolist()))

        self.data.loc[:, 'topic_name'] = self.data['topic'].apply(lambda x: topic_name_map[x])        
        
        self.data.loc[:, 'tf_idf_words'] = self.data['topic'].apply(lambda x: self.topic_model.get_topic(x))
        
        self.data_filename = self.model_directory/ f"data_{my_timestamp}.json"
        self.data.to_json(self.data_filename, orient='records', lines=True)      

        self.write_model_info(my_timestamp)

    def write_model_info(self, my_timestamp = TODAY):
        """This function creates a txt file with information about the model.
        For now these include: keywords, subreddits, topics, and date range.
        """
        
        with open(self.model_directory/ f"INFO_{my_timestamp}.txt", 'w') as the_file:
            #write keywords and subreddits
            for k in ['keyword','subreddit']:
                the_file.write(f"{k.upper()}: {','.join(self.data[k].unique().tolist())}\n")
                the_file.write('\n')
            
            #topics
            the_file.write(f"TOPICS:\n")
            for topic in sorted(self.data['topic_name'].unique().tolist()):
                the_file.write(f"{topic}\n")
            the_file.write('\n')


            #dates range
            dates = pd.to_datetime(self.data['created_utc'])
            min_date = dates.min().date().strftime('%Y-%m-%d')
            max_date = dates.max().date().strftime('%Y-%m-%d')
            the_file.write(f"DATES: {min_date} to {max_date}\n")
            the_file.write('\n')
            
    

    def plot(self, limit_topics = 10, text_column='body', **kwargs):
        '''
        Plots the topic frequency bar chart, the UMAP clusters and (optionally) allows for topic selection to read posts.
 
        Parameters
        ----------
        limit_topics: int
            The number of top topics to display in the bar chart.
        read_posts: bool
            If True, an extra textbox will be added for display of posts belonging to the selected topic.
        text_column: str
            Column in the dataframe to be used for displaying the post content.
        limit_posts: int
            The number of posts to display in the textbox for selected topic.
        Returns
        -------
        altair object of the two charts + (optionally) a text box for display of underlying posts.
        '''

        #TODO: control stuff like width and height through kwargs passed on from streamlit to make the charts more responsive
        data_for_plot = self.data[(self.data['topic']!=-1) & (self.data['topic'] < limit_topics)]
        topicSelection = alt.selection(type="single", encodings=['y'])

        topic_bar = self._plot_topic_frequency_altair(data_for_plot, topicSelection)
        topic_clusters = self._plot_clusters_altair(data_for_plot, topicSelection)

        topic_charts = alt.hconcat(topic_bar, topic_clusters)


        return topic_charts.configure_axis(
                                        labelFontSize=12,
                                        titleFontSize=13,
                                        ) 


    def _get_corpus(self):
        """This function computes the corpus

        Returns:
            list of documents
        """
        return self.data[self.modeling_column].tolist() 


    def _compute_clusters(self):
        '''
        Computes the UMAP clustering of topics.
        '''
        # ideally the TopicModel would have the data and embeddings stored as attributes
        # self.data and self.embeddings so then these methods act directly on them
        # without the need to load them every single time we call one
        X_embedded = UMAP().fit_transform(self.embeddings)
        self.data['dim0'] = X_embedded[:,0]
        self.data['dim1'] = X_embedded[:,1]


    def _plot_clusters_altair(self, data, topic_selector, **kwargs):
        '''
        Plots the UMAP cluster color coded by topic using altair.
        
        Parameters
        ----------
        data: pandas DataFrame
            The dataframe filtered on a subset of the top topics for plotting.
        topic_selector: altair selector object
            A selector object that binds this chart to the topic frequency chart.

        Returns
        -------
        The altair clusters chart.
        '''
        #TODO: figure out colormaps (default ugly AF)
        return alt.Chart(data).mark_circle(size=6).encode(
                                            x=alt.X('dim0:Q', scale=alt.Scale(zero=False)),
                                            y=alt.Y('dim1:Q', scale=alt.Scale(zero=False)),
                                            color='topic:N',
                                            tooltip=['title:N', 'body:N', 'url:N'],
                                            href='url:N').properties(
                                        ).transform_filter(
                                            topic_selector)
    
    def _plot_clusters_plotly(self, data, body_column="body", title_column="title", url_column="url", **kwargs):
        '''
        Plots the UMAP cluster color coded by topic using plotly.
        
        Parameters
        ----------
        data: pandas DataFrame
            The dataframe filtered on a subset of the top topics for plotting.
        topic_selector: altair selector object
            A selector object that binds this chart to the topic frequency chart.

        Returns
        -------
        The altair clusters chart.
        '''
        width = kwargs.pop('width', 800)
        height = kwargs.pop('height', 600)
        return px.scatter(data, x="dim0", y="dim1", color="topic_name", 
                            custom_data=[body_column, title_column, url_column], 
                            width=width, height=height
                            ).update_layout(clickmode='event+select'
                            ).update_traces(marker_size=5)


    def _plot_topic_frequency_altair(self, data, selector, **kwargs):
        '''
        Plot the topic frequency bar chart using altair.

        Parameters
        ----------
        data: pandas DataFrame
            The dataframe filtered on a subset of the top topics for plotting.
        selector: altair selector object
            A selector object that will bind this chart to others (clusters, text, etc.)

        Returns
        -------
        The altair topic frequency chart.
        '''
        return alt.Chart(data).mark_bar().encode(
                                                y=alt.Y('topic_name:N',sort="-x"),
                                                x='count()',
                                                color=alt.condition(selector, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
                                            ).properties(
                                                width=400,
                                                height=300
                                            ).add_selection(selector)


    def _plot_topic_frequency_plotly(self, data, width = 800, height = 600):
        '''
        Plot the topic frequency bar chart using plotlu.

        Parameters
        ----------
        data: pandas DataFrame
            The dataframe filtered on a subset of the top topics for plotting.

        Returns
        -------
        The plotly topic frequency chart.
        '''
        # width = kwargs.pop('width', 800)
        # height = kwargs.pop('height', 600)
        return px.histogram(data, y='topic_name', barmode='group', 
                            width=width, height=height
                            ).update_layout(clickmode='event+select')



    def _plot_textbox(self, data, text_column, topic_selector, limit_posts, **kwargs):
        '''
        Plots the UMAP cluster color coded by topic.
        
        Parameters
        ----------
        data: pandas DataFrame
            The dataframe filtered on a subset of the top topics for plotting.
        text_column: str
            Column in the dataframe to be used for displaying the post content.
        topic_selector: altair selector object
            A selector object that binds this chart to the topic frequency chart.
        limit_posts: int
            The number of posts to display in the textbox (if only topic selected.)

        Returns
        -------
        The altair textbox object.
        '''

        ranked_text = alt.Chart(data).mark_text(align='left',
            dx=-500, size=10).encode(
            y=alt.Y('row_number:O',axis=None)
        ).transform_filter(
            topic_selector
        ).transform_window(
            row_number='row_number()'
        ).transform_window(
            rank='rank(row_number)'
        ).transform_filter( alt.datum.rank<limit_posts).properties(width=1200)

        return ranked_text.encode(text=f'{text_column}:N').properties(title='comment')

    def _plot_topic_frequency_locally(self, limit=10, suffix=None):
        
        #FIXME this will become an altair plot?
        plt.rc('xtick', labelsize=8) 
        plt.rc('ytick', labelsize=8) 
        
        fig = plt.figure(figsize=(5,5))

        self.data[(self.data['topic'] < limit) & 
                  (self.data['topic'] > -1)]['topic_name'].value_counts().plot(kind='barh')
        
        plt.title('topic_name', fontsize=12)
        plt.subplots_adjust(left=0.5)

        if suffix is None:
            suffix = TODAY

        plt.savefig(self.model_directory/ f"topics_freq_{suffix}.pdf")

    def _plot_topic_evolution(self, topics=[]):
        return None

    def load_saved_model_and_data(self, model_timestamp):
        """This function is to load a previously computed model and data

        Args:
            model_timestamp (string): timestamp of when model and data were created
        """

        model_filename = self.model_directory/ f"model_{model_timestamp}"
        embeddings_filename = self.model_directory/ f"embeddings_{model_timestamp}.json"
        self.data_filename = self.model_directory/ f"data_{model_timestamp}.json"

        self.topic_model = BERTopic.load(model_filename)
        self.data = pd.read_json(self.data_filename, lines=True)
        self.embeddings = pd.read_json(embeddings_filename, lines=True)