import re
import warnings

import pandas as pd

import ideo_topic_modeler.utils as ut


class Model:

    def __init__(self, data, text_column, data_source, language='english'):
        '''
        Initializes an instance of the Model class.

        Parameters
        ----------
        data: pandas DataFrame instance
            The dataframe with a column to be used for modeling.
        text_column: str
            The name of the column to be used for topic analysis.
        data_source: str
            where the data are coming from
        '''
        self.language = language
        self.data_source = data_source

        if not data.empty:
            
            self.data = data

            # check if text column exists in data, if not raise an error
            if text_column in self.data.columns:
                self.text_column = text_column
            else:
                raise ValueError(f'{text_column} is not a column of provided dataframe.')

            #added this cause for checks it's useful to keep a copy of the original text.
            self.modeling_column = f"{self.text_column}_clean"

            self.transform_data()
            self.clean_data()

    def transform_data(self):
        """This function transforms the data set, including:
        - dropping missing values
        - shortening the posts to only consider sentences with and around a keyword
        - adding the title to the body text
        """
        
        # Removing data with missing body
        print(f"Size initial dataset = {len(self.data)} rows")
        self.data = self.data.dropna(subset=[self.text_column]).copy()
        print(f"Removing missing values --> {len(self.data)} rows")

        #making a copy of the original text, before cleaning it.
        self.data[f"{self.text_column}_original"] = self.data[self.text_column]

        # For long posts (e.g., Reddit), we only consider the sentence with the keyword and those around it. 
        if self.data_source == 'reddit':
            self.data.loc[:,self.text_column] = self.data[["keyword", self.text_column]].apply(lambda x: Model.return_sentences_around_keyword(x[1], x[0]), 1)
            print(f"Using reddit --> {self.text_column} shortened...")

        #add title to body
        if 'title' in self.data.columns:
            self.data.loc[:,self.text_column] = self.data['title'] + '.' + self.data[self.text_column]

    def clean_data(self):
        """This function contains the cleaning rules
        """

        #decode ascii
        self.data.loc[:, self.modeling_column] = self.data[self.text_column].apply(lambda x: ut.decode_ascii(x))

        #removing https and addresses:
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: re.sub(r"http\S+", "", x, flags=re.I))
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: re.sub(r"www.\S+", "", x, flags=re.I))
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: re.sub("&amp", "", x, flags=re.IGNORECASE))

        #remove some specific punctuation (we want to keep question and exclamation marks)
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: re.sub(r"[\|\.:;@$%_\[\]()+*#\"\/]", ' ', x, flags=re.I))
        
        #remove quotes and apostrophes
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: re.sub("(?<=[a-z])[’'](?=[a-z])", "", x, flags=re.IGNORECASE))

        #remove line breaks and tabs
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: re.sub(r"\s+", ' ', x, flags=re.IGNORECASE))

        #remove trailing spaces
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: x.strip())

        #make text lowercase
        self.data.loc[:, self.modeling_column] = self.data[self.modeling_column].apply(lambda x: x.lower())

        #remove duplicates
        self.data.drop_duplicates(subset=[self.modeling_column], inplace = True)

        #remove data if there are now empty strings
        self.data = self.data[self.data[self.modeling_column]!='']
        
        print(f"Data after cleaning --> {len(self.data)} rows")
            

    @staticmethod
    def return_sentences_around_keyword(text, keyword):
        """This function takes in input text, and it only 
        returns sentences containing the keyword and those around it. 
        
        Args:
            body (str): free text
            keyword (str): keyword of interest

        Returns:
            str: sentences in the original text containing the keyword of interest and the senteces around it
        """
        
        #split the body in a list of sentences
        text_list = text.split('.')  

        #Find the position of the senteces with the keyword of interest
        #For a keyword like 'wellness goal', we do an exact search, for a keyword like 'wellness+goal' we search for either words.
        locations = [n for n, b in enumerate(text_list) if re.search(re.sub('\+', '|', keyword).lower(), b.lower())]

        #collecting only the sentences containing the keyword and those around it, keeping their order.
        body_light = []

        for l in locations:

            #add the sentence before if the keyword is not in the first sentence
            if l > 0:
                body_light.append(text_list[l-1])    
            
            #add the sentence with the keyword
            body_light.append(text_list[l])

            #add the sentence after, if the keyword is not in the last sentence
            if l != len(text_list) - 1:
                body_light.append(text_list[l+1])
            
        #remove duplicates that might have been introduced.
        body_light_dedoup = []
        for b in body_light:
            if b not in body_light_dedoup:
                body_light_dedoup.append(b)

        return '.'.join(body_light_dedoup)


    def filter_data(self):
        # #FIXME add filtering by subreddit if using reddit
        # if self.data_source == 'reddit':
        #     self.data = self.data[self.data['subreddit'].isin(subreddits)]
        return None

    def run(self):
        return None

    def plot(self):
        return None