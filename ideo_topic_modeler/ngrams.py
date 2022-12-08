
import warnings

from ideo_topic_modeler.model import Model


try:
    from nltk.corpus import stopwords as nltk_stopwords
    _use_nltk = True
except:
    _use_nltk = False 
 

class NgramModel(Model):

    def __init__(self, data, text_column, n=1, use_nltk_stopwords=True, custom_stopwords=[], language='english'):
        '''
        Initializes an instance of the n-gram modeling class.

        Parameters
        ----------
        data: pandas DataFrame instance
            The dataframe with a column to be used for n-gram analysis.
        text_column: str
            The name of the column to be used for n-gram analysis.
        n: int
            Number of words to group in an entity. Default is 1.
        use_nltk_stopwords: bool
            Whether to use nltk default stopwords list for the given language (if available). 
            To use, nltk needs to be installed and the stopwords resource downloaded with nltk.download('stopwords').
        custom_stopwords: list
            A list of user-provided words to skip in n-grams.
        '''
        super(NgramModel, self).__init__(data, text_column, language)
        
        # check if n is int, if not convert and raise warning
        if isinstance(n, int):
            self.n = n
        else:
            warnings.warn('Provided n is not an int, converting to nearest int')
            self.n = int(n)
        stopwords = []
        # load stopwords from the specified package
        if use_nltk_stopwords:
            if _use_nltk:
                try:
                    stopwords.extend(nltk_stopwords.words(self.language))
                except Exception as e:
                    raise e
            else:
                raise ImportError('nltk package not installed. Install nltk or set default_stopwords=None')
        if len(custom_stopwords) != 0:
            stopwords.extend(custom_stopwords)
        
        self.stopwords = stopwords

    def run(self):
        '''
        Computes the n-grams, omitting stopwords. 
        
        For n > 1, if a word in the potential n-gram is a stopword, 
        the entire n-gram will not be considered, therefore an option to not use 
        stopwords or use a custom list might be preferable here. 

        Returns
        -------
        ngrams_text: str
            A string of all found n-grams separated by empty space. 
            In n>1, separate words in an n-gram are joined by an underscore.
        '''
        ngrams_text = ""
        for val in self.data[self.text_column].dropna():
            # typecaste each val to string
            val = str(val)
        
            # split the value
            tokens = val.split()
            
            if self.n == 1:
                # only works for unigrams, check if any are stopwords and omit them
                # before adding to the final ngrams string
                tokens_clean = []
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                    if tokens[i] not in self.stopwords:
                        tokens_clean.append(tokens[i])
                    else:
                        pass
                ngrams_text += " ".join(tokens_clean)+" "

            else:
                # for n > 1, we need to make sure that we don't form n-grams with the stopwords
                # therefore each ngram containing a stopword is omitted as a whole - so be mindful with stopwords here!
                ngrams = []
                for i in range(len(tokens)-self.n+1):
                    tokens_n = tokens[i:i+self.n]
                    if any([token_n in self.stopwords for token_n in tokens_n]) or any([token_n == ' ' for token_n in tokens_n]):
                        pass
                    else:
                        ngrams.append("_".join([token.lower() for token in tokens_n]))
                ngrams_text += " ".join(ngrams)+" "

        self.n_grams = ngrams_text 
        return ngrams_text

    def plot(self, type='bar', **kwargs):
        '''
        Plots the n-gram frequency distribution, either as a bar chart or wordcloud.

        Parameters
        ----------
        type: str
            Can be one of ['bar', 'wordcloud']. Determines the type of plot to be produced.
        
        Plot-specific kwargs can also be provided here and will be passed on to the respective methods.

        Returns
        -------
        fig: object
            An altair figure object that can be displayed, saved or embedded in an app.
        '''
        if type=='bar':
            fig = self._plot_bar(**kwargs)
        elif type=='wordcloud':
            fig = self._plot_wordcloud(**kwargs)
        else:
            raise ValueError(f"Unrecognized n-gram plot type {type}. Can be one of ['bar', 'wordcloud']")
        return fig

    def _plot_bar(self, **kwargs):
        '''
        '''
        return None
    
    def _plot_wordcloud(self, **kwargs):
        '''
        '''
        return None