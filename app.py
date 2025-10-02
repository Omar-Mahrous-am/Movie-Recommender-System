import pickle
import pandas as pd
import gradio as gr

# load similarity matrix
with open("cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

# load movies data
df2 = pd.read_csv("movies.csv")
indices = pd.Series(df2.index, index=df2['title']).to_dict()

# recommendation function
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df2['title'].iloc[movie_indices].tolist()
    return pd.DataFrame({"Recommended Movies": recommendations})

# Gradio UI
input_component = gr.Dropdown(choices=df2['title'].tolist(), label="Choose a movie")
output_component = gr.Dataframe()

demo = gr.Interface(
    fn=get_recommendations,
    inputs=input_component,
    outputs=output_component
)

demo.launch(share=True)
