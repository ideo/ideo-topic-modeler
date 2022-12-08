# from pathlib import Path

import pandas as pd

from ideo_topic_modeler import TopicModel, DATA_DIR


if __name__ == "__main__":

    # Get Data  
    # filename = 'reddit_continuous glucose monitoring_in_all-subreddits_2021-01-01_2022-12-01_complete.json'
    # filename = 'reddit_wellness+goals_in_all-subreddits_2022-08-01_2022-12-02_complete.json'
    filename = "reddit_wellness_2022-01-01_2022-12-01_complete.json"
    data = pd.read_json(DATA_DIR / filename, lines=True)

    text_column = 'body'
    data_source = 'reddit'
    start_from_model = None
    # start_from_model = '02_12_2022_114257'


    my_model = TopicModel(data, text_column, data_source, DATA_DIR)
    
    if start_from_model:
        my_model.load_saved_model_and_data(start_from_model)
        my_model._plot_topic_frequency(suffix =start_from_model)
    else:
        my_model.run()
        my_model.save_model()
        my_model.enrich_data_and_save_them()
        my_model._plot_topic_frequency()


