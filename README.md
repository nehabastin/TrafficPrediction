This app is hosted in Render base tier thus the needto upload the model weights to the app
App Deployed URL : https://trafficprediction.onrender.com/
Download model weights here https://drive.google.com/drive/folders/1U-AiYOXukkhTxEDKMse2u4bTlHy_c4XP?usp=sharing


To run the system locally use docker 

Setup Docker 
docker build -t my-streamlit-app .

To Run the app 
docker run -p 8501:8501 my-streamlit-app
